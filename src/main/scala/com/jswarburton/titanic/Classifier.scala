package com.jswarburton.titanic

import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, ProbabilisticClassifier}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.avg

object LogisticRegressionClassifier extends Classifier {
  private val lr = new LogisticRegression()
    .setLabelCol(labelCol)
    .setFeaturesCol(featuresCol)

  def predict(training: DataFrame, test: DataFrame): DataFrame = predict(training, test, lr)
}

object DecisionTreeClassifier extends Classifier {
  private val decisionTree = new DecisionTreeClassifier()
    .setLabelCol(labelCol)
    .setFeaturesCol(featuresCol)

  def predict(training: DataFrame, test: DataFrame): DataFrame = predict(training, test, decisionTree)
}

sealed trait Classifier {
  val featuresCol = "features"
  val labelCol = "Survived"

  private val ageCol = "Age"
  private val fareCol = "Age"

  private val stringCols = Seq("Sex", "Embarked")
  private val numericCols = Seq(ageCol, "SibSp", "Parch", fareCol, "Pclass")

  def predict(training: DataFrame, test: DataFrame): DataFrame

  protected def predict(trainingSet: DataFrame, testSet: DataFrame, classifier: ProbabilisticClassifier[_, _, _]): DataFrame = {
    val model = trainModel(trainingSet, classifier)
    makePredictions(testSet, model)
  }

  private def trainModel(trainingSet: DataFrame, classifier: ProbabilisticClassifier[_, _, _]): PipelineModel = {
    val avgAge = trainingSet.select(avg(ageCol)).first().getDouble(0)
    val NaDefaultValues = Map(ageCol -> avgAge, "Embarked" -> "S")
    val filledTraining = trainingSet.na.fill(NaDefaultValues)

    val stringIndexers = stringCols.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
    }

    val vectorAssembler = new VectorAssembler()
      .setInputCols((numericCols ++ stringCols.map(_ + "Indexed")).toArray)
      .setOutputCol(featuresCol)

    val pipeline = new Pipeline().setStages((stringIndexers :+ vectorAssembler :+ classifier).toArray)

    println("Training...")
    pipeline.fit(filledTraining)
  }

  private def makePredictions(testSet: DataFrame, model: PipelineModel): DataFrame = {
    val avgFare = testSet.select(avg(fareCol)).first().getDouble(0)
    val avgAge = testSet.select(avg(ageCol)).first().getDouble(0)
    val NaDefaultValues = Map(ageCol -> avgAge, "Embarked" -> "S")

    val imputedTestMap = NaDefaultValues + (fareCol -> avgFare)
    val imputedTitanicTest = testSet.na.fill(imputedTestMap)

    println("Predicting...")
    model.transform(imputedTitanicTest)
  }
}

