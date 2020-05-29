val commonSettings = Seq(
  scalaVersion := "2.12.11",
  organization := "io.github.pityka"
)

lazy val jniOsx = project
  .in(file("jni-osx"))
  .settings(commonSettings)
  .settings(
    name := "aten-scala-jni-osx",
    crossPaths := false,
    autoScalaLibrary := false
  )

  lazy val jniLinux = project
  .in(file("jni-linux"))
  .settings(commonSettings)
  .settings(
    name := "aten-scala-jni-linux",
    crossPaths := false,
    autoScalaLibrary := false
  )

lazy val core = project
  .in(file("core"))
  .settings(commonSettings)
  .settings(
    name := "aten-scala-core",
    libraryDependencies ++= Seq(
      "com.github.fommil" % "jniloader" % "1.1"
    )
  )
  .dependsOn(jniOsx)
  .dependsOn(jniLinux)

lazy val root = project
  .in(file("."))
  .settings(
    publishArtifact := false
  )
  .aggregate(jniOsx, core)

lazy val test = project
  .in(file("test"))
  .settings(
    mainClass in Compile := Some("Test"),
    fork := true,
    publishArtifact := false
  )
  .dependsOn(core)
