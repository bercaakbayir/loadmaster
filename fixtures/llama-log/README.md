`llama-server -lv 4` stderr excerpts in the exact formats of llama.cpp commit f174553
(src/llama-model.cpp:2009, src/llama-kv-cache.cpp:291/301, src/llama-context.cpp:468/842,
tools/mtmd/clip.cpp:2197/2206/2256/4003, ggml-cuda.cu:933, ggml-metal-device.m:2160,
tools/server/server.cpp:491/498/525). Numbers are illustrative. Used by the `report` log
parser tests and by `loadmaster __fake-runtime --log-fixture`.
