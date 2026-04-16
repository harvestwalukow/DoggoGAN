import clsx from "clsx";
import { FC, KeyboardEventHandler, useEffect, useState } from "react";
import {
  addWhiteImage,
  generateDog,
  generateImageVariations,
  generatedImagesSlice,
  useAppDispatch,
  useAppSelector,
} from "./state";
import { Content } from "./Content";
import { animated, useSpringRef, useTransition } from "@react-spring/web";
import { Settings } from "./Settings";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { MousePointer2, Hand, Eraser, Wand2, Trash2, PanelLeft } from "lucide-react";


function App() {
  const dispatch = useAppDispatch();
  const editorId = useAppSelector((state) => state.generatedImages.editorId);
  const transRef = useSpringRef();
  const [showSettings, setShowSettings] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const workspaceTool = useAppSelector(
    (state) => state.generatedImages.workspaceTool
  );

  const show = editorId == null;
  const transitionToolbar = useTransition(show, {
    ref: transRef,
    from: { opacity: 0, transform: "translateX(-100px)" },
    enter: { opacity: 1, transform: "translateX(0px)" },
    leave: { opacity: 0, transform: "translateX(-100px)" },
  });

  useEffect(() => {
    transRef.start();
  }, [show, transRef]);

  const onGenerate = (prompt: string) => {
    dispatch(generateImageVariations(prompt, { navigate: true }));
  };
  return (
    <div className="flex-1 relative flex overflow-hidden bg-white">
      {/* Minimalist Sidebar Toggle */}
      <div className="absolute top-4 left-4 z-[60]">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-gray-400 hover:text-gray-600 hover:bg-transparent"
          onClick={() => setShowSidebar(!showSidebar)}
        >
          <PanelLeft className="h-5 w-5" />
        </Button>
      </div>

      {/* Minimalist Sidebar with YouTube Embed */}
      <aside
        className={clsx(
          "fixed left-0 top-0 h-full w-[280px] bg-slate-50/80 backdrop-blur-sm border-r border-gray-100 z-50 transition-all duration-200 ease-out flex flex-col",
          !showSidebar && "-translate-x-full"
        )}
      >
        <div className="flex-1 overflow-y-auto overflow-x-hidden pt-12">
          <div className="p-4">
            <div className="relative w-full aspect-video rounded-lg overflow-hidden bg-gray-100 border border-gray-100 shadow-sm">
              <div className="absolute inset-0 w-[150%] h-[150%] origin-top-left scale-[0.6666]">
                <iframe
                  className="w-full h-full"
                  src="https://www.youtube.com/embed/zuFeyqYKXtw?controls=1&modestbranding=1&rel=0&vq=hd1080"
                  title="DoggoGAN Demo"
                  frameBorder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  allowFullScreen
                ></iframe>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {showSettings && (
        <Settings
          close={() => {
            setShowSettings(false);
          }}
        />
      )}
      <Content />
      {/* Imagine input removed per UI request */}
      {transitionToolbar(
        (style, item) =>
          item && (
            <animated.div style={style} className="absolute bottom-8 w-full flex justify-center pointer-events-none z-50">
              <Card className="canvas-item flex flex-row p-1.5 gap-1 pointer-events-auto items-center shadow-md border-gray-100">
                
                <Tooltip>
                  <TooltipTrigger>
                    <Button
                      variant={workspaceTool === "select-tool" ? "secondary" : "ghost"}
                      className="h-10 w-10 font-bold"
                      onClick={() => dispatch(generatedImagesSlice.actions.setWorkspaceTool({ tool: "select-tool" }))}
                    >
                      <MousePointer2 className="h-5 w-5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top">Select tool</TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger>
                    <Button
                      variant={workspaceTool === "grab-tool" ? "secondary" : "ghost"}
                      className="h-10 w-10"
                      onClick={() => dispatch(generatedImagesSlice.actions.setWorkspaceTool({ tool: "grab-tool" }))}
                    >
                      <Hand className="h-5 w-5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top">Grab tool</TooltipContent>
                </Tooltip>

                <Tooltip>
                  <TooltipTrigger>
                    <Button
                      variant={workspaceTool === "delete-tool" ? "secondary" : "ghost"}
                      className="h-10 w-10"
                      onClick={() => dispatch(generatedImagesSlice.actions.setWorkspaceTool({ tool: "delete-tool" }))}
                    >
                      <Eraser className="h-5 w-5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top">Deletion tool</TooltipContent>
                </Tooltip>

                <Separator orientation="vertical" className="mx-1 h-8 bg-gray-100" />

                <Tooltip>
                  <TooltipTrigger>
                    <Button
                      variant={workspaceTool === "place-tool" ? "secondary" : "ghost"}
                      className="h-10 w-10 text-gray-500 hover:text-gray-900 group"
                      onClick={() => dispatch(generatedImagesSlice.actions.setWorkspaceTool({ tool: "place-tool" }))}
                    >
                      <Wand2 className={clsx("h-5 w-5", workspaceTool === "place-tool" && "text-blue-500")} />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top">Generate Dog</TooltipContent>
                </Tooltip>

                <Separator orientation="vertical" className="mx-1 h-8 bg-gray-100" />

                <Tooltip>
                  <TooltipTrigger>
                    <Button
                      variant="ghost"
                      className="h-10 w-10 text-red-500 hover:bg-red-50 hover:text-red-600"
                      onClick={() => {
                        if (window.confirm("Are you sure you want to delete all images?")) {
                          dispatch(generatedImagesSlice.actions.clearAllImages());
                        }
                      }}
                    >
                      <Trash2 className="h-5 w-5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="top">Clear all images</TooltipContent>
                </Tooltip>

              </Card>
            </animated.div>
          )
      )}
    </div>
  );
}

export default App;
