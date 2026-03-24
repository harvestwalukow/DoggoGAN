import { useSpringRef, animated, useSpring } from "@react-spring/web";
import { RefObject, FC, useState, useEffect, useRef, useMemo } from "react";
import { EclipseHalf } from "react-svg-spinners";
import {
  GeneratedImage,
  useAppDispatch,
  useAppSelector,
  setEditingImage,
  upscaleImage,
  selectUpscaledImageChildren,
  uploadImage,
  generatedImagesSlice,
  generateImageToImageVariations,
  addImageToWorkspace,
} from "./state";
import { assertNever } from "./utils/assertNever";
import { useTransition } from "@react-spring/web";
import FileSave from "file-saver";
import { Loader2, Copy, Download, Trash2, ExternalLink, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import clsx from "clsx";
import { toOptimizedImage } from "./utils/toOptimizedImage";

interface ImagineInputProps {
  id?: string;
  initialText?: string;
  buttonText: string;
  placeholder: string;
  onSubmit: (prompt: string) => void;
}

const ImagineInput: FC<ImagineInputProps> = ({ id, initialText, buttonText, placeholder, onSubmit }) => {
  const [prompt, setPrompt] = useState(initialText || "");
  const [loading, setLoading] = useState(false);

  const submit = () => {
    onSubmit(prompt);
    setPrompt("");
  };

  return (
    <div className="flex w-[350px] items-center space-x-2 bg-white/95 backdrop-blur-sm p-1.5 rounded-2xl shadow-lg border border-gray-100 mb-2 pointer-events-auto">
      <Input
        id={id}
        disabled={loading}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
        placeholder={placeholder}
        autoComplete="off"
        className="border-0 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 px-3 h-10 text-[14px]"
      />
      <Button disabled={loading} onClick={submit} className="h-10 rounded-xl px-4 shrink-0 transition-transform active:scale-95">
        {buttonText}
      </Button>
    </div>
  );
};

interface ImageEditorProps {
  image: GeneratedImage;
  isEditing: boolean;
  imageRef: RefObject<HTMLImageElement>;
}
export const Image: FC<ImageEditorProps> = ({ imageRef, image, isEditing }) => {
  const dispatch = useAppDispatch();
  const inTransRef = useSpringRef();

  const [tool, setTool] = useState<"generate-variations" | null>(null);
  const imageChildren = useAppSelector((state) =>
    selectUpscaledImageChildren(state, image.id)
  );
  const [activePosition, setActivePosition] = useState<number | null>(null);

  useEffect(() => {
    if (!isEditing && tool != null) {
      setTool(null);
    }
  }, [isEditing, tool]);

  const transitionsEditorRight = useTransition(
    isEditing,
    useMemo(
      () => ({
        ref: inTransRef,
        from: { opacity: 0, transform: "translateX(-50px)" },
        enter: { opacity: 1, transform: "translateX(0px)" },
        leave: { opacity: 0, transform: "translateX(-50px)" },
      }),
      [inTransRef]
    )
  );

  useEffect(() => {
    inTransRef.start();
  }, [isEditing, inTransRef]);

  useEffect(() => {
    if (!isEditing) {
      setActivePosition(null);
    }
  }, [isEditing]);

  const initialText = useMemo(() => {
    if (image.prompt === "white background") {
      return "";
    }
    return image.prompt.replace(/(?:https?|ftp):\/\/[\n\S]+/g, "");
  }, [image.prompt]);

  if (image.url == null) return null;

  const activePositionAction:
    | "none"
    | "jump-to-image"
    | "upscale-image"
    | "loading-image" = (() => {
    if (activePosition == null) {
      return "none";
    }
    if (
      activePosition in imageChildren &&
      imageChildren[activePosition].percentageDone === 100
    ) {
      return "jump-to-image";
    }
    if (activePosition in imageChildren) {
      return "loading-image";
    }
    return "upscale-image";
  })();

  const onGenerate = (prompt: string) => {
    if (image.type === "upscaled") {
      dispatch(generateImageToImageVariations(image.id, 0, prompt));
      return;
    }
    if (activePosition == null) {
      console.error("No active position");
      return;
    }
    dispatch(generateImageToImageVariations(image.id, activePosition, prompt));
  };

  return (
    <>

      {transitionsEditorRight((style, item) => {
        return (
          item && (
            <animated.div style={style} className="absolute left-full ml-6 top-1/2 z-[100]">
              <div className="-translate-y-1/2">
                 <Editor
                  position={activePosition}
                  imageState={activePositionAction}
                  image={image}
                  tool={tool}
                  setTool={setTool}
                  initialText={initialText}
                  onGenerate={onGenerate}
                />
              </div>
            </animated.div>
          )
        );
      })}
      {(() => {
        if (image.url == null) {
          return null;
        }
        if (typeof image.url === "string") {
          return (
            <img
              ref={imageRef}
              className="w-[64px] h-[64px] select-none"
              style={{ imageRendering: "pixelated" }}
              src={toOptimizedImage(image.url)}
            />
          );
        }

        return (
          <div
            className={clsx(
              "flex flex-col gap-2 w-[64px] h-[64px]"
            )}
          >
            {image.url.map((url, position) => {
              const img = (
                <img
                  key={position}
                  ref={imageRef}
                  className="w-[64px] h-[64px] select-none"
                  style={{ imageRendering: "pixelated" }}
                  src={toOptimizedImage(url)}
                />
              );
              if (!isEditing) {
                return img;
              }
              return (
                <div key={position} className="relative">
                  {img}
                  <div
                    className={clsx(
                      "absolute left-0 top-0 w-full h-full p-2 transition-all",
                      activePosition === position
                        ? "shadow-lg shadow-black z-10"
                        : "border-transparent",
                      activePosition != null &&
                        activePosition !== position &&
                        "bg-[rgba(0,0,0,0.4)]"
                    )}
                    onClick={() => {
                      setActivePosition((p) =>
                        p === position ? null : position
                      );
                    }}
                    onDoubleClick={() => {
                      if (imageChildren[position] == null) return;
                      dispatch(setEditingImage(imageChildren[position].id));
                    }}
                  />
                </div>
              );
            })}
          </div>
        );
      })()}
    </>
  );
};



interface EditorProps {
  position: number | null;
  imageState: "jump-to-image" | "upscale-image" | "loading-image" | "none";
  image: GeneratedImage;
  tool: "generate-variations" | null;
  setTool: (tool: "generate-variations" | null) => void;
  initialText: string;
  onGenerate: (prompt: string) => void;
}
const Editor: FC<EditorProps> = ({
  imageState,
  position,
  image,
  tool,
  setTool,
  initialText,
  onGenerate,
}) => {
  const dispatch = useAppDispatch();
  const imageChildren = useAppSelector((state) =>
    selectUpscaledImageChildren(state, image.id)
  );
  const [loadingSave, setLoadingSave] = useState(false);

  return (
    <Card className="flex flex-col h-fit w-56 text-[12px] cursor-auto select-none overflow-hidden shadow-md p-0 gap-0">
      <div className="px-4 py-2 font-medium text-xs border-b flex items-center justify-between">
        <span>Editor</span>
        {image.classId !== undefined && (
          <span className="text-[10px] text-gray-400 font-normal">Breed #{image.classId}</span>
        )}
      </div>
      <div className="flex flex-col flex-1 text-gray-800 py-1">
        {position == null && image.type === "variations" && (
          <div className="basis-1/4 text-gray-700 flex justify-center items-center mx-2">
            Click on an image
          </div>
        )}
        {!(position == null && image.type === "variations") && (
          <Button
            variant="ghost"
            className={clsx(
              "w-full justify-start rounded-none px-4 py-2.5 h-auto text-xs font-normal",
              tool === "generate-variations" && "bg-gray-100"
            )}
            onClick={() => {
              if (image.type === "upscaled" && image.isCanvas) {
                window.alert(
                  "Only use black and white colors when generating an image based of a doodle!s"
                );
              }
              setTool(
                tool === "generate-variations" ? null : "generate-variations"
              );
              setTimeout(
                () => document.getElementById("variations-input")?.focus(),
                100
              );
            }}
          >
            <Copy className="mr-2.5 h-3.5 w-3.5 text-gray-400 group-hover:text-blue-500 transition-colors" />
            <span className="font-medium">Generate variations</span>
          </Button>
        )}

        {tool === "generate-variations" && (
          <div className="px-4 py-2 border-b bg-gray-50/50">
            <div className="flex flex-col gap-2">
              <Input
                id="variations-input"
                autoFocus
                defaultValue={initialText}
                placeholder="Variant prompt..."
                className="h-8 text-[11px] bg-white border-gray-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    onGenerate((e.target as HTMLInputElement).value);
                    setTool(null);
                  }
                }}
              />
              <div className="flex justify-end gap-2">
                <Button 
                  size="sm" 
                  variant="ghost" 
                  className="h-6 text-[10px] px-2"
                  onClick={() => setTool(null)}
                >
                  Cancel
                </Button>
                <Button 
                  size="sm" 
                  className="h-6 text-[10px] px-3 bg-blue-600 hover:bg-blue-700"
                  onClick={() => {
                    const input = document.getElementById("variations-input") as HTMLInputElement;
                    onGenerate(input.value);
                    setTool(null);
                  }}
                >
                  Generate
                </Button>
              </div>
            </div>
          </div>
        )}

        {image.type === "variations" &&
          position != null &&
          imageState !== "none" && (
            <Button
              variant="ghost"
              className="w-full justify-start rounded-none px-4 py-2.5 h-auto text-xs font-normal"
              onClick={() => {
                switch (imageState) {
                  case "loading-image":
                  case "jump-to-image": {
                    dispatch(setEditingImage(imageChildren[position].id));
                    return;
                  }
                  case "upscale-image": {
                    dispatch(addImageToWorkspace(image.id, position));
                    return;
                  }
                  default: {
                    assertNever(imageState);
                  }
                }
              }}
            >
              {(() => {
                switch (imageState) {
                  case "loading-image":
                  case "jump-to-image": {
                    return <ExternalLink className="mr-2.5 h-3.5 w-3.5 text-gray-500" />;
                  }
                  case "upscale-image": {
                    return <Sparkles className="mr-2.5 h-3.5 w-3.5 text-gray-500" />;
                  }
                  default: {
                    assertNever(imageState);
                  }
                }
              })()}
              {(() => {
                switch (imageState) {
                  case "loading-image":
                  case "jump-to-image": {
                    return "Go to image";
                  }
                  case "upscale-image": {
                    return "Add to canvas";
                  }
                  default: {
                    assertNever(imageState);
                  }
                }
              })()}
            </Button>
          )}
        {image.type === "upscaled" && (
          <Button
            variant="ghost"
            className="w-full justify-start rounded-none px-4 py-2.5 h-auto text-xs font-normal disabled:opacity-80 text-gray-700"
            onClick={() => {
              if (image.url == null) {
                return;
              }
              FileSave.saveAs(image.url[0], "image.png");
            }}
          >
            {loadingSave ? (
              <Loader2 className="mr-2.5 h-3.5 w-3.5 text-gray-400 animate-spin" />
            ) : (
              <Download className="mr-2.5 h-3.5 w-3.5 text-gray-400 group-hover:text-green-500 transition-colors" />
            )}
            <span className="font-medium">Download image</span>
          </Button>
        )}
        {image.type !== "variations" && (
          <Button
            variant="ghost"
            className="w-full justify-start rounded-none px-4 py-2.5 h-auto text-xs font-normal text-red-600 hover:text-red-700 hover:bg-red-50"
            onClick={() => {
              dispatch(
                generatedImagesSlice.actions.deleteImage({ id: image.id })
              );
            }}
          >
            <Trash2 className="mr-2.5 h-3.5 w-3.5 text-red-400 group-hover:text-red-600 transition-colors" />
            <span className="font-medium">Delete image</span>
          </Button>
        )}
      </div>
    </Card>
  );
};


