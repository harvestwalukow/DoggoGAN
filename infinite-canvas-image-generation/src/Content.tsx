import {
  FC,
  MouseEventHandler,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  GeneratedImage,
  generateDog,
  generatedImagesSlice,
  localWorkspaceTransform,
  onWorkspaceElement,
  setEditingImage,
  smoothTransformWorkspace,
  transformWorkspace,
  useAppDispatch,
  useAppSelector,
} from "./state";
import React from "react";
import clsx from "clsx";
import { animated, useSpring } from "@react-spring/web";
import { batchActions } from "redux-batched-actions";
import { Image } from "./ImageEditor";
import { Loader2 } from "lucide-react";

const MAX_ZOOM_STEP = 10;
const IS_DARWIN = /Mac|iPod|iPhone|iPad/.test(
  typeof window === "undefined" ? "node" : window.navigator.platform
);

function normalizeWheel(event: WheelEvent | React.WheelEvent<HTMLElement>) {
  let { deltaY, deltaX } = event;
  let deltaZ = 0;

  if (event.ctrlKey || event.altKey || event.metaKey) {
    const signY = Math.sign(event.deltaY);
    const absDeltaY = Math.abs(event.deltaY);

    let dy = deltaY;

    if (absDeltaY > MAX_ZOOM_STEP) {
      dy = MAX_ZOOM_STEP * signY;
    }

    deltaZ = dy / 100;
  } else {
    if (event.shiftKey && !IS_DARWIN) {
      deltaX = deltaY;
      deltaY = 0;
    }
  }

  return { x: -deltaX, y: -deltaY, z: -deltaZ };
}

export const Content: FC = () => {
  const dispatch = useAppDispatch();
  const images = useAppSelector((state) => state.generatedImages.images);
  const editorId = useAppSelector((state) => state.generatedImages.editorId);
  const workspaceImages = useAppSelector(
    (state) => state.generatedImages.workspaceImages
  );
  const activeImageId = useAppSelector(
    (state) => state.generatedImages.activeImageId
  );
  const workspaceTool = useAppSelector((state) => state.generatedImages.workspaceTool);

  const [dragging, setDragging] = useState(false);
  const [mouseCanvasPos, setMouseCanvasPos] = useState({ x: 0, y: 0 });
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const workspaceRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (activeImageId == null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Delete") {
        dispatch(
          generatedImagesSlice.actions.deleteImage({ id: activeImageId })
        );
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
    };
  }, [activeImageId, dispatch]);

  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();

      const delta = normalizeWheel(e);
      const isMoving =
        delta.x !== 0 && (Math.abs(delta.x) > 2 || Math.abs(delta.y) > 2);
      if (
        editorId != null &&
        (isMoving || localWorkspaceTransform.scale < 0.9)
      ) {
        dispatch(setEditingImage(null));
      }

      if (delta.z === 0) {
        dispatch(
          transformWorkspace({
            y: localWorkspaceTransform.y + delta.y,
            x: localWorkspaceTransform.x + delta.x,
          })
        );

        return;
      }
      const currentScale = localWorkspaceTransform.scale;
      const scaling = ((currentScale - 0.1) * (3 - 1)) / (5 - 0.1) + 1;
      const scaleBy = delta.z * scaling;
      const newScale = Math.min(
        Math.min(Math.max(currentScale + scaleBy * 1.3, 0.1), 4),
        5
      );
      if (newScale === currentScale) return;

      const target = e.currentTarget as HTMLElement;
      const fromTop = e.clientY / target.clientHeight - 0.5;
      const fromLeft = e.clientX / target.clientWidth - 0.5;

      const transformY =
        (-fromTop * scaleBy * target.clientHeight) / currentScale;
      const transformX =
        (-fromLeft * scaleBy * target.clientWidth) / currentScale;

      const deltaX =
        (localWorkspaceTransform.x * (newScale - currentScale)) / currentScale +
        transformX;
      const deltaY =
        (localWorkspaceTransform.y * (newScale - currentScale)) / currentScale +
        transformY;

      dispatch(
        transformWorkspace({
          x: localWorkspaceTransform.x + deltaX,
          y: localWorkspaceTransform.y + deltaY,
          scale: newScale,
        })
      );
    };

    const target = viewportRef.current;
    if (target == null) return;
    target.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      target.removeEventListener("wheel", handleWheel);
    };
  }, [dispatch, editorId]);

  const onMouseDown: MouseEventHandler = (e) => {
    if (workspaceTool === "place-tool") {
      dispatch(generateDog({ x: mouseCanvasPos.x - 32, y: mouseCanvasPos.y - 32 }));
      dispatch(generatedImagesSlice.actions.setWorkspaceTool({ tool: "select-tool" }));
      return;
    }
    if (e.target !== e.currentTarget) return;
    setDragging(true);
    dispatch(setEditingImage(null));
    dispatch(
      batchActions([
        generatedImagesSlice.actions.setActiveImage({ id: null }),
        generatedImagesSlice.actions.setEditorImageId({ id: null }),
      ])
    );
  };

  const onMouseMove: MouseEventHandler = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const canvasX =
      (mouseX - rect.width / 2 - localWorkspaceTransform.x) /
      localWorkspaceTransform.scale;
    const canvasY =
      (mouseY - rect.height / 2 - localWorkspaceTransform.y) /
      localWorkspaceTransform.scale;

    setMouseCanvasPos({ x: canvasX, y: canvasY });

    if (dragging) {
      dispatch(
        transformWorkspace({
          y: localWorkspaceTransform.y + e.movementY,
          x: localWorkspaceTransform.x + e.movementX,
        })
      );
    }
  };

  const onMouseUp: MouseEventHandler = () => {
    setDragging(false);
  };

  useEffect(() => {
    if (workspaceRef.current == null) return;
    dispatch(onWorkspaceElement(workspaceRef.current));
  }, [dispatch]);

  return (
    <div
      ref={viewportRef}
      className={clsx(
        "relative flex-1",
        workspaceTool === "place-tool" ? "cursor-crosshair" : "cursor-default"
      )}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
    >

      <div
        ref={workspaceRef}
        id="workspace"
        style={{
          // transform: `translate(${workspaceTransform.x}px, ${workspaceTransform.y}px) scale(${workspaceTransform.scale})`,
          transformOrigin: "50% 50%",
        }}
        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 will-change-transform"
      >
        {useMemo(() => {
          return Object.values(images).map((image) => (
            <GeneratedImageItem key={image.id} image={image} />
          ));
        }, [images])}

        {workspaceTool === "place-tool" && (
          <div
            className="absolute border-2 border-dashed border-blue-400 w-[64px] h-[64px] pointer-events-none rounded-sm"
            style={{
              transform: `translate(${mouseCanvasPos.x - 32}px, ${
                mouseCanvasPos.y - 32
              }px)`,
            }}
          />
        )}
      </div>
    </div>
  );
};

interface GeneratedImageItemProps {
  image: GeneratedImage;
}
const GeneratedImageItem: FC<GeneratedImageItemProps> = ({ image }) => {
  const dispatch = useAppDispatch();

  const editorId = useAppSelector((state) => state.generatedImages.editorId);
  const isActiveImage = useAppSelector(
    (state) => state.generatedImages.activeImageId === image.id
  );
  const scale = useAppSelector(
    (state) => state.generatedImages.workspaceTransform.scale
  );
  const workspaceTool = useAppSelector(
    (state) => state.generatedImages.workspaceTool
  );

  const imageRef = useRef<HTMLImageElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const isEditing = editorId === image.id;
  const isOtherImageEditing = !isEditing && editorId != null;
  const style = useSpring(
    useMemo(
      () => ({
        opacity: isOtherImageEditing ? 0 : 1,
        // immediate,
      }),
      [isOtherImageEditing]
    )
  );

  const onMouseDown: MouseEventHandler = (e) => {
    if (isEditing || workspaceTool === "delete-tool") {
      return;
    }
    e.preventDefault();
    dispatch(generatedImagesSlice.actions.setActiveImage({ id: image.id }));

    const onMouseMove = (e: MouseEvent) => {
      setDragging(true);
      dispatch(
        generatedImagesSlice.actions.moveImage({
          id: image.id,
          transform: {
            x: e.movementX / scale,
            y: e.movementY / scale,
          },
        })
      );
    };
    const onMouseUp = () => {
      setDragging(false);
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    };
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);
  };

  const onClick = () => {
    if (workspaceTool === "delete-tool") {
      dispatch(generatedImagesSlice.actions.deleteImage({ id: image.id }));
      return;
    }
    dispatch(
      generatedImagesSlice.actions.setActiveImage({
        id: image.id,
      })
    );
    if (
      image.url == null ||
      isEditing ||
      dragging ||
      workspaceTool !== "select-tool"
    ) {
      return;
    }
    dispatch(setEditingImage(image.id));
  };

  const cursor = (() => {
    if (workspaceTool === "delete-tool") {
      return "cursor-not-allowed";
    }
    if (dragging) {
      return "cursor-grabbing";
    }
    if (workspaceTool === "grab-tool" || image.url == null) {
      return "cursor-grab";
    }
    return "cursor-pointer";
  })();

  return (
    <animated.div
      style={{
        transform: `translate(${
          isActiveImage && !isEditing
            ? image.transform.x - 1
            : image.transform.x
        }px, ${
          isActiveImage && !isEditing
            ? image.transform.y - 1
            : image.transform.y
        }px)`,
        ...style,
      }}
      className={clsx(
        `absolute cursor- box-border flex top-0 left-0 translate-x-1/2 translate-y-1/2 transition-shadow canvas-item rounded-sm z-10`,
        isEditing && "z-20",
        cursor
      )}
      onMouseUp={onClick}
      onMouseDown={onMouseDown}
    >
      {image.url != null ? (
        <Image imageRef={imageRef} image={image} isEditing={isEditing} />
      ) : (
        <div className="w-[64px] h-[64px] bg-gray-100 flex justify-center items-center flex-col gap-y-1">
          <Loader2 className="h-4 w-4 animate-spin text-gray-400" />
          <div className="text-gray-600 text-[6px] break-words px-1 flex-1 justify-center grow-0 max-w-full leading-tight hidden">
            {image.prompt}
          </div>
        </div>
      )}
    </animated.div>
  );
};
