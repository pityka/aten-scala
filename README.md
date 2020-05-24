# Build documentation 
1. needs system wide libtorch e.g. brew install libtorch
2. needs bloop and sbt
3. `make prepare` - this creates bloop build definitions from sbt
4. `make test` 
5. `make publishLocal`

The Makefile will generate Java and C++ sources and create a jni native library.

`parser/` holds a parser for `parseable.h` which is a subset of the `libtorch/include/ATen/Functions.h`. This parser generates cpp (`./wrapper.cpp`) and java JNI code (`./aten-scala/core/src/main/java/ATen.java`). The cpp code is compiled to `./aten-scala/jni-osx/src/main/resources/libatenscalajni.dylib`.

`aten-scala/` holds an sbt project with two subprojects. `aten-scala/core` is the Java counterpart of the bnding. It depends on Scala because `std::tuple` is translated into Scala tuples.  `aten-scala/jni-osx` contains no source code but its Maven artifact contains the native library.

# How to use
See `aten-scala/test/`.





