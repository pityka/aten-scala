[![maven](https://img.shields.io/maven-central/v/io.github.pityka/aten-scala-core_2.13.svg)](https://repo1.maven.org/maven2/io/github/pityka/aten-scala-core_2.13/)

# Build documentation 

The JNI bindings are built with clang++ for Mac and for Linux. 
The build runs on a Mac and uses Docker for cross compilation. 
The docker image used for building (and possible at runtime) is defined in `docker-runtime/Dockerfile`. This image is pushed to the Docker Hub (https://hub.docker.com/r/pityka/base-ubuntu-libtorch/tags?page=1&ordering=last_updated), and by default the build will pull it. 

## Prerequisites
- needs two macs with clang++ installed: one with intel, one with M1. Cross compilation did not work.
- needs docker on the mac 
- on the both mac needs the libtorch shared libraries in /usr/local/lib/. Read on.
- needs bloop (https://scalacenter.github.io/bloop/) and sbt (https://www.scala-sbt.org/).


## Build
1. `make prepare` - this creates bloop build definitions from sbt the sbt build definition
2. `make libatenscalajni_x86.dylib` - on intel
3. `make libatenscalajni_aarm64.dylib` - on M1
4. Copy from one to the other 
5. `make test`
6. Copy `aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib` to the other mac, `make test` there as well.
3. `make test-linux` - builds for linux and runs tests in a linux container (needs docker for mac, will pull `pityka/base-ubuntu-libtorch` from docker hub)
6. `make test-cuda` - runs tests in a cuda enabled remote docker context


## Publishing
- `make publishLocal` - publishes artifacts locally
- `make publish` - publishes artifacts to github packages. 
- `bash publish.sh` - publishes the artifacts to Maven Central

# How this works
The Makefile will generate Java and C++ sources and create a JNI native library.

`parser/` holds a parser for `parseable.h` which is a subset of the `libtorch/include/ATen/Functions.h`. This parser generates cpp (`./wrapper.cpp`) and java JNI code (`./aten-scala/core/src/main/java/ATen.java`). The cpp code is compiled to `./aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib`.

`aten-scala/` holds an sbt project with two subprojects. `aten-scala/core` is the Java counterpart of the binding. It depends on Scala because `std::tuple` is translated into Scala tuples.  `aten-scala/jni-osx` contains no source code but its Maven artifact contains the native library.

# Libtorch runtime dependency
Libtorch is linked dynamically during runtime. 
## OSX
On OSX the jni shared library always looks for libtorch in `/usr/local/lib/`.
You can get the libtorch libraries with `pip3 install torch` then copy the necessary files to /usr/local/lib/. E.g. cp `/usr/local/lib/python3.9/site-packages/torch/lib/* /usr/local/lib/`. 
However one needs to copy `/usr/local/lib/python3.9/site-packages/torch/.dylibs/libomp.dylib` as well to `/usr/local/.dylibs/` (depends on the libtorch version).
## Linux
On Linux there is no @rpath baked in the jni shared library. 
Use LD_LIBRARY_PATH to make sure the linker finds libtorch.
See the `docker-runtime/Dockerfile`s on how to do this.

# Docker image for Linux with libtorch and CUDA
See `docker-runtime/Dockerfile`. This image is pushed to the Docker Hub (https://hub.docker.com/r/pityka/base-ubuntu-libtorch/tags?page=1&ordering=last_updated).

The image also has Java 8 and sbt. It is ready to make use of this library on Linux.

# License
See LICENSE file.





