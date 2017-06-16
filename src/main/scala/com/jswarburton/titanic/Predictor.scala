package com.jswarburton.titanic

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}

object Predictor {
  private val spark = SparkSession.builder.
    master("local[1]")
    .appName("Titanic Predictor")
    .getOrCreate()

  // Datasets
  private val trainingSet = "src/main/resources/train.csv"
  private val testSet = "src/main/resources/test.csv"

  def main(args: Array[String]): Unit = {
    val (training, test) = loadTrainingAndTestSets(trainingSet, testSet)

    val dtPredictions = DecisionTreeClassifier.predict(training, test)
    outputPredictions("decision-tree-output", dtPredictions)

    val lrPredictions = LogisticRegressionClassifier.predict(training, test)
    outputPredictions("logistic-regression-output", lrPredictions)

    spark.stop()
  }

  private def loadTrainingAndTestSets(trainFile: String, testFile: String): (DataFrame, DataFrame) = {
    import spark.implicits._

    val training = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(trainFile)
      .withColumn("Survived", $"Survived".cast(DoubleType))
      .cache()

    val test = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(testFile)
      .cache()

    println("Training set schema:")
    training.printSchema()

    println("Test set schema:")
    test.printSchema()

    (training, test)
  }

  private def outputPredictions(outputPrefix: String, predictions: DataFrame) = {
    import spark.implicits._
    val outputPath = s"src/main/resources/$outputPrefix"
    predictions
      .select($"PassengerId", $"prediction".cast(IntegerType).alias("Survived"))
      .coalesce(1)
      .write
      .option("header", "true")
      .csv(outputPath)
  }
}
