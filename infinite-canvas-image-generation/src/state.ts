import {
  Action,
  AnyAction,
  PayloadAction,
  ThunkAction,
  combineReducers,
  configureStore,
  createSelector,
  createSlice,
} from "@reduxjs/toolkit";
import { TypedUseSelectorHook, useDispatch, useSelector } from "react-redux";
import { batchActions, enableBatching } from "redux-batched-actions";
import { persistReducer, persistStore } from "redux-persist";
import storage from "redux-persist/es/storage";
import { v4 } from "uuid";
import { assertNever } from "./utils/assertNever";
import { findEmptyArea } from "./utils/findEmptyArea";
import { throttle } from "lodash-es";
import { REPLICATE_TOKEN_KEY } from "./Settings";
import { uploadFile } from "@uploadcare/upload-client";
import { getApi } from "./utils/getApi";

const initialTransform = { x: 0, y: 0, scale: 1 };

type ImageChild =
  | { type: "upscaled"; id: string; position?: number }
  | { type: "variations"; id: string };

type VariationGeneration = {
  type: "variations";
  id: string;
  url: string[] | null;
  prompt: string;
  percentageDone: number;
  parent: {
    id: string;
    position: number;
  } | null;
  children: ImageChild[];
  transform: {
    x: number;
    y: number;
  };
  classId?: number;
};
type UpscaledGeneration = {
  type: "upscaled";
  id: string;
  url: [string] | null; // stay consistent cause I'm lazy but still get typesafety
  prompt: string;
  percentageDone: number;
  isCanvas: boolean;
  parent: {
    id: string;
    position: number;
  } | null;
  children: ImageChild[];
  transform: {
    x: number;
    y: number;
  };
  classId?: number;
};
export type GeneratedImage = VariationGeneration | UpscaledGeneration;

type HistoryItem =
  | { type: "image-editor"; imageId: string }
  | {
      type: "workspace-transform";
      transform: { x: number; y: number; scale: number };
    };

interface GeneratedImagesState {
  images: Record<string, GeneratedImage>;
  editorId: string | null;
  prevEditorId: string | null;
  activeImageId: string | null;
  workspaceTool: "select-tool" | "grab-tool" | "delete-tool" | "place-tool";
  workspaceTransform: { x: number; y: number; scale: number };
  workspaceImages: Record<string, true>;
  history: HistoryItem[];
  historyIndex: number;
}
// prefilled with some images
// Empty initial state
const initialState: GeneratedImagesState = {
  images: {},
  editorId: null,
  prevEditorId: null,
  activeImageId: null,
  workspaceTool: "select-tool",
  workspaceTransform: initialTransform,
  workspaceImages: {},
  history: [],
  historyIndex: -1,
};

export const generatedImagesSlice = createSlice({
  name: "generatedImagesSlice",
  initialState,
  reducers: {
    setWorkspaceTool: (
      state,
      action: PayloadAction<{ tool: GeneratedImagesState["workspaceTool"] }>
    ) => {
      state.workspaceTool = action.payload.tool;
    },
    addImage: (state, action: PayloadAction<GeneratedImage>) => {
      state.images[action.payload.id] = action.payload;
    },
    showImageInWorkspace: (state, action: PayloadAction<{ id: string }>) => {
      state.workspaceImages[action.payload.id] = true;
    },
    hideImageInWorkspace: (state, action: PayloadAction<{ id: string }>) => {
      delete state.workspaceImages[action.payload.id];
    },
    deleteImage: (state, action: PayloadAction<{ id: string }>) => {
      delete state.images[action.payload.id];
      if (state.editorId === action.payload.id) {
        state.editorId = null;
      }
      if (state.activeImageId === action.payload.id) {
        state.activeImageId = null;
      }
      state.history = state.history.filter(
        (item) =>
          item.type !== "image-editor" || item.imageId !== action.payload.id
      );
    },
    clearAllImages: (state) => {
      state.images = {};
      state.editorId = null;
      state.activeImageId = null;
      state.history = [];
      state.historyIndex = -1;
    },
    setWorkspaceTransform: (
      state,
      action: PayloadAction<{ x?: number; y?: number; scale?: number }>
    ) => {
      if (action.payload.x != null) {
        state.workspaceTransform.x = action.payload.x;
      }
      if (action.payload.y != null) {
        state.workspaceTransform.y = action.payload.y;
      }
      if (action.payload.scale != null) {
        state.workspaceTransform.scale = action.payload.scale;
      }
    },
    setImageUrls: (
      state,
      action: PayloadAction<{ id: string; urls: string[]; classId?: number }>
    ) => {
      const image = state.images[action.payload.id];
      if (image == null) {
        console.error("Invalid image", action.payload.id);
        return;
      }
      image.url = action.payload.urls;
      if (action.payload.classId !== undefined) {
        image.classId = action.payload.classId;
      }
    },
    appendImageChild: (
      state,
      action: PayloadAction<{ id: string; child: ImageChild }>
    ) => {
      state.images[action.payload.id]!.children.push(action.payload.child);
    },
    removeImageChild: (
      state,
      action: PayloadAction<{ id: string; childId: string }>
    ) => {
      const index = state.images[action.payload.id]!.children.findIndex(
        (child) => child.id === action.payload.childId
      );
      if (index === -1) {
        return;
      }
      state.images[action.payload.id]!.children.splice(index, 1);
    },
    setImagePercentage: (
      state,
      action: PayloadAction<{ id: string; percentage: number }>
    ) => {
      state.images[action.payload.id]!.percentageDone =
        action.payload.percentage;
    },
    moveImage: (
      state,
      action: PayloadAction<{ id: string; transform: { x: number; y: number } }>
    ) => {
      state.images[action.payload.id]!.transform.x +=
        action.payload.transform.x;
      state.images[action.payload.id]!.transform.y +=
        action.payload.transform.y;
    },
    setEditorImageId: (state, action: PayloadAction<{ id: string | null }>) => {
      state.prevEditorId = state.editorId;
      state.editorId = action.payload.id;
    },
    setActiveImage: (state, action: PayloadAction<{ id: string | null }>) => {
      state.activeImageId = action.payload.id;
    },
    appendHistory: (state, action: PayloadAction<HistoryItem>) => {
      const isRefocus = (() => {
        if (
          state.history.length === 0 ||
          action.payload.type !== "image-editor"
        ) {
          return false;
        }
        const lastItem = state.history[state.history.length - 1];
        if (lastItem.type !== "image-editor") {
          return false;
        }
        return lastItem.imageId === action.payload.imageId;
      })();
      if (isRefocus) {
        return;
      }
      state.history.splice(state.historyIndex + 1);
      state.history.push(action.payload);
      state.historyIndex = state.history.length - 1;
    },
    navigateHistory: (
      state,
      action: PayloadAction<{ historyIndex: number }>
    ) => {
      state.historyIndex = action.payload.historyIndex;
    },
  },
});

const getEditorScale = () => {
  const referenceHeight = 1000;
  const referenceWidth = 1500;
  const scaleAtReference = 1.6;
  const scaleHeight = (window.innerHeight / referenceHeight) * scaleAtReference;
  const scaleWidth = (window.innerWidth / referenceWidth) * scaleAtReference;
  return Math.round(Math.min(scaleHeight, scaleWidth) * 10) / 10;
};

const debounceTransformWorkspace = throttle(
  (dispatch, transform) => {
    dispatch(generatedImagesSlice.actions.setWorkspaceTransform(transform));
  },
  50,
  { leading: true, trailing: true }
);

// this state is not reactive cause just to expensive, instead it is synced with workspaceEl every time it's set
export const localWorkspaceTransform: { x: number; y: number; scale: number } =
  { ...initialTransform };

export const transformWorkspace = (
  transform: {
    x?: number;
    y?: number;
    scale?: number;
  },
  immediate?: boolean
): AppThunk => {
  return async (dispatch, _getState, extra) => {
    if (extra.workspaceEl == null) return;
    if (transform.scale != null) {
      localWorkspaceTransform.scale = transform.scale;
    }
    if (transform.x != null) {
      localWorkspaceTransform.x = transform.x;
    }
    if (transform.y != null) {
      localWorkspaceTransform.y = transform.y;
    }
    extra.workspaceEl.style.transform = `translate(${localWorkspaceTransform.x}px, ${localWorkspaceTransform.y}px) scale(${localWorkspaceTransform.scale})`;
    if (immediate) {
      dispatch(generatedImagesSlice.actions.setWorkspaceTransform(transform));
      return;
    }
    debounceTransformWorkspace(dispatch, transform);
  };
};

export const navigateHistory = (offset: number): AppThunk => {
  return async (dispatch, getState) => {
    const state = getState().generatedImages;

    const historyIndex = (() => {
      // NOTE(gab): navigates to the latest image that was edited if currently not editing
      if (offset === -1 && state.editorId == null) {
        return state.historyIndex;
      }
      return Math.min(
        Math.max(state.historyIndex + offset, 0),
        state.history.length - 1
      );
    })();

    dispatch(
      generatedImagesSlice.actions.navigateHistory({
        historyIndex,
      })
    );
    const action = state.history[historyIndex];
    switch (action.type) {
      case "image-editor": {
        dispatch(setEditingImage(action.imageId, { keepHistory: true }));
        return;
      }
      case "workspace-transform": {
        dispatch(transformWorkspace(action.transform, true));
        return;
      }
      default: {
        assertNever(action);
      }
    }
  };
};

export const setEditingImage = (
  imageId: string | null,
  opts?: { keepHistory: boolean }
): AppThunk => {
  return async (dispatch, getState, extra) => {
    dispatch(
      generatedImagesSlice.actions.setActiveImage({
        id: imageId,
      })
    );
    if (imageId == null) {
      dispatch(
        generatedImagesSlice.actions.setEditorImageId({
          id: imageId,
        })
      );
      return;
    }

    const actions: AnyAction[] = [
      generatedImagesSlice.actions.setEditorImageId({
        id: imageId,
      }),
    ];
    if (opts?.keepHistory !== true) {
      actions.push(
        generatedImagesSlice.actions.appendHistory({
          type: "image-editor",
          imageId,
        })
      );
    }

    dispatch(batchActions(actions));

    const image = getState().generatedImages.images[imageId];
    const newScale = getEditorScale();
    dispatch(
      smoothTransformWorkspace({
        x: (-image.transform.x - 64 / 2) * newScale,
        y: (-image.transform.y - 64 / 2) * newScale,
        scale: newScale,
      })
    );
  };
};

export const smoothTransformWorkspace = (transform: {
  x?: number;
  y?: number;
  scale?: number;
}): AppThunk => {
  return async (dispatch, _getState, extra) => {
    extra.workspaceEl!.style.transition = "transform 400ms ease-in-out";
    dispatch(transformWorkspace(transform, true));
    setTimeout(() => {
      extra.workspaceEl!.style.transition = "none";
    }, 420);
  };
};

export const uploadImage = async (blob: Blob): Promise<string> => {
  const result = await uploadFile(blob, {
    publicKey: "037a51a72cf85bf758c7",
    store: true,
    metadata: {},
  });

  return result["cdnUrl"] as string;
};

export const addWhiteImage = (): AppThunk => {
  return async (dispatch, getState) => {
    const state = getState();
    const workspaceTransform = state.generatedImages.workspaceTransform;
    const images = state.generatedImages.images;

    const image: GeneratedImage = {
      id: v4(),
      url: ["https://ucarecdn.com/1b9e1cef-ed30-450d-a88f-ded57eb6ec35/"],
      type: "upscaled",
      percentageDone: 100,
      prompt: "white background",
      isCanvas: true,
      parent: null,
      children: [],
      transform: findEmptyArea(
        -workspaceTransform.x * (1 / workspaceTransform.scale),
        -workspaceTransform.y * (1 / workspaceTransform.scale),
        Object.values(images)
      ),
    };
    dispatch(generatedImagesSlice.actions.addImage(image));
    const newScale = Math.max(1.2, workspaceTransform.scale);
    dispatch(
      smoothTransformWorkspace({
        x: (-image.transform.x - 400 / 2) * newScale,
        y: (-image.transform.y - 400 / 2) * newScale,
        scale: newScale,
      })
    );
  };
};

export const generateImageVariations = (
  prompt: string,
  opts?: { navigate?: boolean }
): AppThunk => {
  return async (dispatch, getState) => {
    const workspaceTransform = getState().generatedImages.workspaceTransform;

    if (prompt === "empty") {
      dispatch(addWhiteImage());
      return;
    }

    const tmpId = v4();
    const image: GeneratedImage = {
      type: "variations",
      id: tmpId,
      url: null,
      percentageDone: 100,
      prompt,
      parent: null,
      children: [],
      transform: findEmptyArea(
        -workspaceTransform.x / workspaceTransform.scale,
        -workspaceTransform.y / workspaceTransform.scale,
        Object.values(getState().generatedImages.images)
      ),
    };
    dispatch(batchActions([generatedImagesSlice.actions.addImage(image)]));
    if (opts?.navigate) {
      const scale = getState().generatedImages.workspaceTransform.scale;

      const newScale = Math.max(1.1, scale);
      dispatch(
        smoothTransformWorkspace({
          x: (-image.transform.x - 400 / 2) * newScale,
          y: (-image.transform.y - 400 / 2) * newScale,
          scale: newScale,
        })
      );
    }

    let imageUrls: string[] = [];
    try {
      const response = await fetch(`${getApi()}/imagine-variations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt,
          replicateToken: localStorage.getItem(REPLICATE_TOKEN_KEY),
        }),
      });
      if (response.status < 200 || response.status >= 300) {
        console.error(
          "Could not generate image",
          response.status,
          response.statusText,
          response
        );
      }
      const result = await response.json();
      imageUrls = result["variations"];
    } catch (e) {
      console.error(e);
      return;
    }

    dispatch(
      batchActions([
        generatedImagesSlice.actions.setImageUrls({
          id: tmpId,
          urls: imageUrls,
        }),
        generatedImagesSlice.actions.setImagePercentage({
          id: tmpId,
          percentage: 100,
        }),
      ])
    );
  };
};

export const generateImageToImageVariations = (
  imageId: string,
  position: number,
  prompt: string
): AppThunk => {
  return async (dispatch, getState) => {
    const state = getState();
    const originalImage = state.generatedImages.images[imageId];
    const workspaceTransform = state.generatedImages.workspaceTransform;
    const images = state.generatedImages.images;
    if (originalImage == null) {
      console.error("Could not find image", imageId);
      return;
    }

    // Breed-aware variations for dogs
    if (originalImage.classId !== undefined) {
      const image: GeneratedImage = {
        id: v4(),
        url: null,
        type: "variations",
        percentageDone: 0,
        prompt: `Variations of breed #${originalImage.classId}`,
        parent: { id: originalImage.id, position },
        children: [],
        transform: findEmptyArea(
          -workspaceTransform.x * (1 / workspaceTransform.scale),
          -workspaceTransform.y * (1 / workspaceTransform.scale),
          Object.values(images)
        ),
        classId: originalImage.classId,
      };

      dispatch(
        batchActions([
          generatedImagesSlice.actions.addImage(image),
          generatedImagesSlice.actions.appendImageChild({
            id: originalImage.id,
            child: { id: image.id, type: "variations" },
          }),
        ])
      );

      try {
        const response = await fetch(`${getApi()}/generate-dog`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            classId: originalImage.classId,
            numImages: 4,
          }),
        });

        if (response.status < 200 || response.status >= 300) {
          console.error("Could not generate dog variations", response.status);
          return;
        }

        const result = await response.json();
        dispatch(
          batchActions([
            generatedImagesSlice.actions.setImageUrls({
              id: image.id,
              urls: result.variations,
              classId: result.class_id,
            }),
            generatedImagesSlice.actions.setImagePercentage({
              id: image.id,
              percentage: 100,
            }),
          ])
        );
      } catch (e) {
        console.error(e);
      }
      return;
    }

    if (originalImage.url == null) {
      console.error("Could not find image URL", imageId);
      return;
    }

    const image: GeneratedImage = {
      id: v4(),
      url: null,
      type: "upscaled",
      isCanvas: false,
      percentageDone: 0,
      prompt,
      parent: { id: originalImage.id, position },
      children: [],
      transform: findEmptyArea(
        -workspaceTransform.x * (1 / workspaceTransform.scale),
        -workspaceTransform.y * (1 / workspaceTransform.scale),
        Object.values(images)
      ),
    };

    dispatch(
      batchActions([
        generatedImagesSlice.actions.addImage(image),
        generatedImagesSlice.actions.appendImageChild({
          id: originalImage.id,
          child: { id: image.id, type: "variations" },
        }),
      ])
    );

    const imageUrls = await (async () => {
      if (originalImage.type === "upscaled" && originalImage.isCanvas) {
        const response = await fetch(`${getApi()}/sketch-to-image-variations`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt,
            url: originalImage.url![position],
            replicateToken: localStorage.getItem(REPLICATE_TOKEN_KEY),
          }),
        });
        if (response.status < 200 || response.status >= 300) {
          console.error(
            "Could not generate image",
            response.status,
            response.statusText,
            response
          );
        }
        const result = await response.json();
        return result["variations"];
      }

      const response = await fetch(`${getApi()}/image-to-image-variations`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt,
          url: originalImage.url![position],
          replicateToken: localStorage.getItem(REPLICATE_TOKEN_KEY),
        }),
      });
      if (response.status < 200 || response.status >= 300) {
        console.error(
          "Could not generate image",
          response.status,
          response.statusText,
          response
        );
      }
      const result = await response.json();
      return result["variations"];
    })();

    dispatch(
      batchActions([
        generatedImagesSlice.actions.setImageUrls({
          id: image.id,
          urls: imageUrls,
        }),
      ])
    );
  };
};

export const upscaleImage = (imageId: string, position: number): AppThunk => {
  return async (dispatch, getState) => {
    const originalImage = getState().generatedImages.images[imageId];
    if (originalImage == null || originalImage.url == null) {
      console.error("Could not find image", imageId);
      return;
    }

    const image: GeneratedImage = {
      id: v4(),
      url: null,
      type: "upscaled",
      percentageDone: 0,
      isCanvas: false,
      parent: { id: imageId, position },
      prompt: originalImage.prompt,
      children: [],
      transform: {
        x:
          originalImage.transform.x +
          Math.floor((Math.random() - 0.5) * 200 + 900),
        y: originalImage.transform.y + Math.floor((Math.random() - 0.5) * 500),
      },
    };
    dispatch(
      batchActions([
        generatedImagesSlice.actions.addImage(image),
        generatedImagesSlice.actions.appendImageChild({
          id: imageId,
          child: { type: "upscaled", id: image.id, position },
        }),
      ])
    );

    let imageUrl: string;
    try {
      const response = await fetch(`${getApi()}/upscale`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url: originalImage.url[position],
          replicateToken: localStorage.getItem(REPLICATE_TOKEN_KEY),
        }),
      });
      if (response.status < 200 || response.status >= 300) {
        console.error(
          "Could not generate image",
          response.status,
          response.statusText,
          response
        );
      }
      const result = await response.json();
      imageUrl = result["upscaled"];
    } catch (e) {
      console.error(e);
      return;
    }

    dispatch(
      batchActions([
        generatedImagesSlice.actions.setImageUrls({
          id: image.id,
          urls: [imageUrl],
        }),
      ])
    );
  };
};


export const generateDog = (pos?: { x: number; y: number }): AppThunk => {
  return async (dispatch, getState) => {
    const state = getState();
    const images = state.generatedImages.images;
    const workspaceTransform = state.generatedImages.workspaceTransform;

    const image: GeneratedImage = {
      id: v4(),
      url: null,
      type: "upscaled", 
      percentageDone: 0,
      isCanvas: false,
      parent: null,
      prompt: "AI Image",
      children: [],
      transform: pos ?? findEmptyArea(
        -workspaceTransform.x / workspaceTransform.scale,
        -workspaceTransform.y / workspaceTransform.scale,
        Object.values(images)
      ),
    };

    dispatch(generatedImagesSlice.actions.addImage(image));

    try {
      const response = await fetch(`${getApi()}/generate-dog`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      });

      if (response.status < 200 || response.status >= 300) {
        console.error(
            "Could not generate dog",
            response.status,
            response.statusText
        );
        return;
      }
      
      const result = await response.json();
      const imageUrls = result.variations;

      if (imageUrls && imageUrls.length > 0) {
        dispatch(
            generatedImagesSlice.actions.setImageUrls({
            id: image.id,
            urls: imageUrls,
            classId: result.class_id,
            })
        );
      }
    } catch (e) {
      console.error(e);
    }
  };
};

export const addImageToWorkspace = (
  imageId: string,
  position: number
): AppThunk => {
  return (dispatch, getState) => {
    const state = getState();
    const originalImage = state.generatedImages.images[imageId];
    const workspaceTransform = state.generatedImages.workspaceTransform;
    const images = state.generatedImages.images;
    if (originalImage == null || originalImage.url == null) {
      console.error("Could not find image", imageId);
      return;
    }

    const image: GeneratedImage = {
      id: v4(),
      url: [originalImage.url[position]],
      type: "upscaled",
      percentageDone: 0,
      isCanvas: false,
      parent: { id: imageId, position },
      prompt: originalImage.prompt,
      children: [],
      transform: findEmptyArea(
        -workspaceTransform.x / workspaceTransform.scale,
        -workspaceTransform.y / workspaceTransform.scale,
        Object.values(images)
      ),
    };
    dispatch(
      batchActions([
        generatedImagesSlice.actions.addImage(image),
        generatedImagesSlice.actions.appendImageChild({
          id: imageId,
          child: { type: "upscaled", id: image.id, position },
        }),
      ])
    );
  };
};

export const selectUpscaledImageChildren = createSelector(
  (state: RootState) => state.generatedImages.images,
  (_: RootState, imageId: string) => imageId,
  (images, imageId): Record<number, GeneratedImage> => {
    const image = images[imageId];
    const loadingImages: Record<number, GeneratedImage> = {};
    for (const child of image.children) {
      const childImage = images[child.id];
      if (
        child.type !== "upscaled" ||
        child.position == null ||
        childImage == null
      ) {
        continue;
      }
      loadingImages[child.position] = childImage;
    }
    return loadingImages;
  }
);

export const onWorkspaceElement = (workspaceEl: HTMLElement): AppThunk => {
  return (_dispatch, _getState, extra) => {
    extra.workspaceEl = workspaceEl;
  };
};

const persistConfigGeneratedImage = {
  key: "generatedImages",
  storage,
  blacklist: ["transform", "workspaceTool"],
  throttle: 1000,
};

interface ThunkExtra {
  workspaceEl: HTMLElement | null;
}
const thunkExtra: ThunkExtra = {
  workspaceEl: null,
};

export const store = configureStore({
  reducer: enableBatching(
    combineReducers({
      generatedImages: persistReducer(
        persistConfigGeneratedImage,
        generatedImagesSlice.reducer
      ),
    })
  ),

  middleware: (getDefaultMiddlware) => {
    return getDefaultMiddlware({
      thunk: {
        extraArgument: thunkExtra,
      },
    });
  },
  devTools: import.meta.env.DEV,
});

export const persistor = persistStore(store);

export type RootState = {
  generatedImages: ReturnType<typeof generatedImagesSlice.reducer>;
};

export type AppDispatch = typeof store.dispatch;
export type AppThunk<R = void> = ThunkAction<
  R,
  RootState,
  ThunkExtra,
  Action<string>
>;
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
