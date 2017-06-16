name := "titanic-ml"

version := "1.0"

scalaVersion := "2.11.8"

val sparkVersion = "2.1.1"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-mllib_2.11" % sparkVersion,
  "org.apache.spark" % "spark-core_2.11" % sparkVersion
)
    