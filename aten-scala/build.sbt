val RELEASE_TO_SONATYPE = sys.env.getOrElse("RELEASE_SONATYPE", "false").toBoolean 

ThisBuild / dynverSonatypeSnapshots := true

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

pomExtra in Global := {
  <url>https://pityka.github.io/aten-scala</url>
  <licenses>
    <license>
      <name>MIT</name>
      <url>https://opensource.org/licenses/MIT</url>
    </license>
  </licenses>
  <scm>
    <connection>scm:git:github.com/pityka/aten-scala</connection>
    <developerConnection>scm:git:git@github.com:pityka/aten-scala</developerConnection>
    <url>github.com/pityka/aten-scala</url>
  </scm>
  <developers>
    <developer>
      <id>pityka</id>
      <name>Istvan Bartha</name>
      <url>https://pityka.github.io/aten-scala/</url>
    </developer>
  </developers>
}
