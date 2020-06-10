# Build documentation 
0. needs a mac with clang++ installed
1. needs system wide libtorch e.g. brew install libtorch
2. needs bloop and sbt
3. `make prepare` - this creates bloop build definitions from sbt
4. `make test` - runs tests on mac. Tests pass if the suite does not crash or throws exception.
5. `make test-linux` - runs tests in a linux container (needs docker for mac)
6. `make test-cuda` - runs tests in a cuda enabled remote docker context
7. `make publishLocal` - publishes artifacts locally
8. `make publish` - publishes artifacts to github packages

The Makefile will generate Java and C++ sources and create a jni native library.

`parser/` holds a parser for `parseable.h` which is a subset of the `libtorch/include/ATen/Functions.h`. This parser generates cpp (`./wrapper.cpp`) and java JNI code (`./aten-scala/core/src/main/java/ATen.java`). The cpp code is compiled to `./aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib`.

`aten-scala/` holds an sbt project with two subprojects. `aten-scala/core` is the Java counterpart of the bnding. It depends on Scala because `std::tuple` is translated into Scala tuples.  `aten-scala/jni-osx` contains no source code but its Maven artifact contains the native library.

# Libtorch dependency
Libtorch is linked dynamically. On OSX you can `brew install libtorch`. 

On linux you have to install torch with pip then add the relevant folders to the ld search path. 
See the `Dockerfile`s on how to do this.

# License
See LICENSE file.





