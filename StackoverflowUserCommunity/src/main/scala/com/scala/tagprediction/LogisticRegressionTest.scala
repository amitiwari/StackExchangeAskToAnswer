package com.scala.tagprediction

import scala.xml._
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.StopWordsRemover
import scala.util.matching.Regex
import scala.io.Source._
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector
import collection.mutable.HashMap
import collection.mutable._
import org.apache.spark.Accumulator
object LogisticRegressionTest {
 	def main(args:Array[String]) = {
 	  val conf = new SparkConf().setAppName("LG").setMaster("local");
		val sc= new SparkContext(conf)
		
		
		Logger.getLogger("org").setLevel(Level.OFF)
		Logger.getLogger("akka").setLevel(Level.OFF)
    val modelDirectory = "/home/neel/lrmodels"
    val testfileName = "/home/neel/datascience.stackexchange.com/Test.xml"
    val textFile = sc.textFile(testfileName)
    val stopwordList = fromFile("/home/neel/twitter/en.txt").getLines.toArray
    val postsXml = textFile.map(_.trim).filter(!_.startsWith("<?xml version=")).filter(_ != "<posts>").filter(_ != "</posts>")
    val numFeatures = 50000
    val result = new StringBuilder
  	val postsRDD = postsXml.map { s =>
          		val xml = XML.loadString(s)
          
          		val id = (xml \ "@Id").text
          		val postidtype = (xml \ "@PostTypeId").text
          		val tags = (xml \ "@Tags").text
          
          		val title = (xml \ "@Title").text
          		val body = (xml \ "@Body").text
          		val bodyPlain = ("<\\S+>".r).replaceAllIn(body, " ")
          		val text = (title + " " + bodyPlain).replaceAll("\n", " ").replaceAll("( )+", " ").replaceAll("\\W", " ");
          
          		Row(id, postidtype,tags, text)
  		}
		
		val postsarray = postsRDD.collect();
		val postSchemaString = "Id PostTypeId Tags Text"
		val postSchema = StructType(postSchemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))

		val sqlContext = new SQLContext(sc)
	
		val postsDf = sqlContext.createDataFrame(postsRDD, postSchema).registerTempTable("postsDf")
		val questionsRDD = sqlContext.sql("SELECT * from postsDf where PostTypeId=1")
    
		
		for( i <- new java.io.File(modelDirectory).listFiles){
      val myudf: (String => Double) = (str: String) => {if (str.contains(i.getName)) 1.0 else 0.0}
  	  val sqlfunc = udf(myudf)
  		val postsLabeled = questionsRDD.withColumn("Label", sqlfunc(col("Tags")))
  		
  		val postsLabeledtokenized = LogisticRgressionLBFGS.tokenize(postsLabeled, "Text", "Words")
  		val postsLabeledtokenizedAndStopWordRemoved = LogisticRgressionLBFGS.removeStopWords(postsLabeledtokenized, "Words", "clenaedText", stopwordList)
  		val postLabeledVectorized = LogisticRgressionLBFGS.createVector(postsLabeledtokenizedAndStopWordRemoved, "clenaedText", "Features", numFeatures)
  			
  		val  test = postLabeledVectorized.rdd.map { 
  	  	      row => LabeledPoint(row.getDouble(4), row(7).asInstanceOf[org.apache.spark.mllib.linalg.Vector])}
		  
      val model = LogisticRegressionModel.load(sc, i.toString())
      val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
            val prediction = model.predict(features)
            if(prediction==1.0){
              result++=i.getName()
            }
            (prediction, label)
      }
     
      val temp_r = predictionAndLabels.collect();
      for(j <- temp_r){
        if(j._1==1.0){
          println(i.getName)
        }
      }
        }
		
 	}
 	 
}