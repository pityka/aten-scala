val RELEASE_TO_SONATYPE = sys.env.getOrElse("RELEASE_SONATYPE", "false").toBoolean 

inThisBuild(
  List(
    organization := "io.github.pityka",
    homepage := Some(url("https://pityka.github.io/aten-scala/")),
    licenses := List(("MIT", url("https://opensource.org/licenses/MIT"))),
    developers := List(
      Developer(
        "pityka",
        "Istvan Bartha",
        "bartha.pityu@gmail.com",
        url("https://github.com/pityka/aten-scala")
      )
    ),
    parallelExecution := false,
    scmInfo := Some(
      ScmInfo(
        url("https://pityka.github.io/aten-scala/"),
        "scm:git@github.com:pityka/aten-scala.git"
      )
    )
  )
)




val commonSettings = Seq(
  scalaVersion := "2.13.12",
  organization := "io.github.pityka",
  licenses += ("MIT", url("http://opensource.org/licenses/MIT")),
  publishTo := sonatypePublishTo.value
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
    crossScalaVersions := Seq("2.13.8","2.12.15","3.3.1"),
    name := "aten-scala-core",
    libraryDependencies ++= Seq(
      "com.github.fommil" % "jniloader" % "1.1"
    )
  )
  .dependsOn(jniOsx)
  .dependsOn(jniLinux)

lazy val root = project
  .in(file("."))
  .settings(commonSettings)
  .settings(
    publishArtifact := false,
    publish / skip := true
  )
  .aggregate(jniOsx, jniLinux, core)

lazy val test = project
  .in(file("test"))
  .settings(commonSettings)
  .settings(
     Compile / mainClass := Some("Test"),
    fork := true,
    publishArtifact := false,
    publish / skip := true
  )
  .dependsOn(core)

publishArtifact := false

publish / skip := true

// pomExtra in Global := {
//   <developers>
//     <developer>
//       <id>pityka</id>
//       <name>Istvan Bartha</name>
//       <url>https://pityka.github.io/utils-string/</url>
//     </developer>
//   </developers>
// }