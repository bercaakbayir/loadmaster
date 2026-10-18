# loadmaster on top of llama.cpp's official server image.
#   docker build -f packaging/docker/Dockerfile --build-arg BASE=server-cuda -t loadmaster:cuda .
#   docker build -f packaging/docker/Dockerfile --build-arg BASE=server      -t loadmaster:cpu  .
#   docker run --gpus all -p 8080:8080 -v /srv/models:/models loadmaster:cuda
# The lockfile is read from /models/models.lock; its proxy.listen must be 0.0.0.0:8080
# inside a container, and its sources must be paths under /models.
ARG BASE=server-cuda

FROM rust:1.85-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p loadmaster

FROM ghcr.io/ggml-org/llama.cpp:${BASE}
COPY --from=build /src/target/release/loadmaster /usr/local/bin/loadmaster
# The base image ships llama-server at /app; make it resolvable as `llama-server`.
ENV PATH="/app:${PATH}" LOADMASTER_HOME=/var/lib/loadmaster
VOLUME ["/models", "/var/lib/loadmaster"]
EXPOSE 8080
ENTRYPOINT ["loadmaster"]
CMD ["up", "--lock", "/models/models.lock"]
