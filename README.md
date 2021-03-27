[![maven](https://img.shields.io/maven-central/v/io.github.pityka/aten-scala-core_2.13.svg)](https://repo1.maven.org/maven2/io/github/pityka/aten-scala-core_2.13/)

# Build documentation 

The JNI bindings are built with clang++ for Mac and for Linux. 
The build runs on a Mac and uses Docker for cross compilation. 
The docker image used for building (and possible at runtime) is defined in `docker-runtime/Dockerfile`. This image is pushed to the Docker Hub (https://hub.docker.com/r/pityka/base-ubuntu-libtorch/tags?page=1&ordering=last_updated), and by default the build will pull it. 

## Prerequisites
- needs a mac with clang++ installed
- needs docker on the mac
- on the mac needs system wide libtorch. There are multiple methods to install libtorch. One is to copy the *.dylib files from the libtorch distribution of the appropriate pytorch version (e.g. https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip) to /usr/local/lib/ . One other method to install libtorch on mac is via brew.
- needs bloop (https://scalacenter.github.io/bloop/) and sbt (https://www.scala-sbt.org/).

## Build
1. `make prepare` - this creates bloop build definitions from sbt the sbt build definition
2. `make test` - builds the binding for mac, and runs tests. Tests pass if the suite does not crash or throws exception.
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
Libtorch is linked dynamically during runtime. On OSX you can `brew install libtorch`, or by copying the .dylib files from the libtorch distribution archive to /usr/local/lib/. 
On Linux you can install torch with pip then add the relevant folders to the ld search path. 
See the `docker-runtime/Dockerfile`s on how to do this.

# Docker image for Linux with libtorch and CUDA
See `docker-runtime/Dockerfile`. This image is pushed to the Docker Hub (https://hub.docker.com/r/pityka/base-ubuntu-libtorch/tags?page=1&ordering=last_updated).

The image also has Java 8 and sbt. It is ready to make use of this library on Linux.

# License
See LICENSE file.





