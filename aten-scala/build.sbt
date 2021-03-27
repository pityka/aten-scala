val RELEASE_TO_SONATYPE = sys.env.getOrElse("RELEASE_SONATYPE", "false").toBoolean 

developers := List(
  Developer(id="pityka", name="Istvan Bartha", email="bartha.pityu@gmail.com", url=url("http://pityka.github.io"))
)

val commonSettings = Seq(
  scalaVersion := "2.13.5",
  organization := "io.github.pityka",
  licenses += ("MIT", url("http://opensource.org/licenses/MIT")),
  githubOwner := "pityka",
  githubRepository := "aten-scala",
  githubTokenSource := TokenSource.GitConfig("github.token") || TokenSource.Environment("GITHUB_TOKEN"),
  publishTo := (if (RELEASE_TO_SONATYPE) sonatypePublishTo.value else sbtghpackages.GitHubPackagesKeys.githubPublishTo.value)
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
  .settings(commonSettings)
  .settings(
    publishArtifact := false,
    skip in publish := true
  )
  .aggregate(jniOsx, jniLinux, core)

lazy val test = project
  .in(file("test"))
  .settings(commonSettings)
  .settings(
    mainClass in Compile := Some("Test"),
    fork := true,
    publishArtifact := false,
    skip in publish := true
  )
  .dependsOn(core)

publishArtifact := false

skip in publish := true