package com.scala.tagprediction
import scala.xml._
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.StopWordsRemover
import scala.util.matching.Regex
import scala.io.Source._

object TagPredict {

	def main(args:Array[String]) = {
		Logger.getLogger("org").setLevel(Level.OFF)
		Logger.getLogger("akka").setLevel(Level.OFF)
		val conf = new SparkConf().setAppName("Naive Bayes").setMaster("local");
		val sc= new SparkContext(conf)
		val fileName = "/home/neel/datascience.stackexchange.com/Posts.xml"
		val tagfileName = "/home/neel/datascience.stackexchange.com/Tags.xml"
		val stopwords = fromFile("/home/neel/twitter/en.txt").getLines.toArray
		val textFile = sc.textFile(fileName)
		val tagsFile = sc.textFile(tagfileName)

		val postsXml = textFile.map(_.trim).filter(!_.startsWith("<?xml version=")).filter(_ != "<posts>").filter(_ != "</posts>")
		val tagsXml = tagsFile.map(_.trim).filter(!_.startsWith("﻿<?xml version=")).filter(_ != "<tags>").filter(_ != "</tags>")

  		val postsRDD = postsXml.map { s =>
      		val xml = XML.loadString(s)
      
      		val id = (xml \ "@Id").text
      		val tags = (xml \ "@Tags").text
      
      		val title = (xml \ "@Title").text
      		val body = (xml \ "@Body").text
      		val bodyPlain = ("<\\S+>".r).replaceAllIn(body, " ")
      		val text = (title + " " + bodyPlain).replaceAll("\n", " ").replaceAll("( )+", " ").replaceAll("\\W", " ");
      
      		Row(id, tags, text)
  		}
		val tagsRDD = tagsXml.map { s =>
    		val xml = XML.loadString(s)
    		val id = (xml \ "@Id").text
    		val tags = (xml \ "@TagName")    
    		val count = (xml \ "@Count")
    		Row(id, tags, count)
		}

		val taglist = tagsRDD.map { x => x(1)}.collect()

		val postSchemaString = "Id Tags Text"
		val postSchema = StructType(postSchemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))

		val sqlContext = new SQLContext(sc)
	
		val postsDf = sqlContext.createDataFrame(postsRDD, postSchema)
		
		println(taglist.length)
	//	for(i <- 0 until taglist.length){
		    val targetTag = "bigdata"
		    val myudf: (String => Double) = (str: String) => {if (str.contains(targetTag)) 1.0 else 0.0}
	  	  val sqlfunc = udf(myudf)
				val postsLabeled = postsDf.withColumn("Label", sqlfunc(col("Tags")) )
				
				val positive = postsLabeled.filter( postsLabeled("Label") >0.0)
        val negative = postsLabeled.filter(postsLabeled("Label") < 1.0)
        
        // Sample without replacement (false)
        val positiveTrain = positive.sample(false, 0.9)
        val negativeTrain = negative.sample(false, 0.9)
        val training = positiveTrain.unionAll(negativeTrain)


				//// CREATE MODEL
				val numFeatures = 64000
				val numEpochs = 30
				val regParam = 0.02

				val tokenizer = new Tokenizer().setInputCol("Text").setOutputCol("Words")
				  val tokenized = tokenizer.transform(training)
				val remover = new StopWordsRemover().setInputCol("Words").setOutputCol("StopWordsRemoved").setStopWords(stopwords)
				val cleanedText = remover.transform(tokenized)
				
	  	  val hashingTF = new  org.apache.spark.ml.feature.HashingTF().setNumFeatures(numFeatures).
          setInputCol("StopWordsRemoved").setOutputCol("Features")
        val lr = new LogisticRegression().setMaxIter(numEpochs).setRegParam(regParam).
                                    setFeaturesCol("Features").setLabelCol("Label").
                                    setRawPredictionCol("Score").setPredictionCol("Prediction")
        val pipeline = new Pipeline().setStages(Array(hashingTF, lr))

        val model = pipeline.fit(cleanedText)
        
        val testingResult = model.transform(cleanedText)
        val testingResultScores = testingResult.select("Prediction", "Label").rdd.
                                    map(r => (r(0).asInstanceOf[Double], r(1).asInstanceOf[Double]))
                                     
        testingResultScores.foreach{x=>
          if(x._1!=x._2)
            println(x)
	  	  }
        val bc = new BinaryClassificationMetrics(testingResultScores)
        val roc = bc.areaUnderROC 
        print("Area under the ROC:" + roc)
	}
}