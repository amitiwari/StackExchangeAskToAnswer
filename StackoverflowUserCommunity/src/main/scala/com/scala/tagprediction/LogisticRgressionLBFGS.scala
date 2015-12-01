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
object LogisticRgressionLBFGS {
	def main(args:Array[String]) = {
		Logger.getLogger("org").setLevel(Level.OFF)
		Logger.getLogger("akka").setLevel(Level.OFF)
		val conf = new SparkConf().setAppName("Train LG").setMaster("local");
		val sc= new SparkContext(conf)
		val fileName = "/home/neel/datascience.stackexchange.com/Posts.xml"
		val tagfileName = "/home/neel/datascience.stackexchange.com/Tags.xml"
		val stopwordList = fromFile("/home/neel/twitter/en.txt").getLines.toArray
		val textFile = sc.textFile(fileName)
		val tagsFile = sc.textFile(tagfileName)
		val numFeatures = 50000
		val numEpochs = 30
		val regParam = 0.02

		val postsXml = textFile.map(_.trim).filter(!_.startsWith("<?xml version=")).filter(_ != "<posts>").filter(_ != "</posts>")
		val tagsXml = tagsFile.map(_.trim).filter(!_.startsWith("﻿<?xml version=")).filter(_ != "<tags>").filter(_ != "</tags>")

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
		val tagsRDD = tagsXml.map { s =>
    		val xml = XML.loadString(s)
    		val id = (xml \ "@Id").text
    		val tags = (xml \ "@TagName")    
    		val count = (xml \ "@Count")
    		Row(id, tags, count)
		}

		val taglist  = tagsRDD.map { x => x(1).toString()}.collect()
		val postSchemaString = "Id PostTypeId Tags Text"
		val postSchema = StructType(postSchemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
		val sqlContext = new SQLContext(sc)

		val postsDf = sqlContext.createDataFrame(postsRDD, postSchema).registerTempTable("postsDf")
		val questionsRDD = sqlContext.sql("SELECT * from postsDf where PostTypeId=1")

		for(i <- taglist){
			        val targetTag = i
		          val myudf: (String => Double) = (str: String) => {if (str.contains(targetTag)) 1.0 else 0.0}
		        	val sqlfunc = udf(myudf)
							val postsLabeled = questionsRDD.withColumn("Label", sqlfunc(col("Tags")))
							val postsLabeledtokenized = tokenize(postsLabeled, "Text", "Words")
							val postsLabeledtokenizedAndStopWordRemoved = removeStopWords(postsLabeledtokenized, "Words", "clenaedText", stopwordList)
							val postLabeledVectorized = createVector(postsLabeledtokenizedAndStopWordRemoved, "clenaedText", "Features", numFeatures)
							val  traning = postLabeledVectorized.rdd.map { 
						        row => LabeledPoint(row.getDouble(4), row(7).asInstanceOf[org.apache.spark.mllib.linalg.Vector])
					    }
					    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(traning)
							//  Compute raw scores on the test set.
							model.save(sc, "/home/neel/lrmodels/" + i)

		}
	}
	def tokenize(inputDataframe : DataFrame, inputCol : String, outputCol : String) : DataFrame ={
		val tokenizer = new Tokenizer().setInputCol(inputCol).setOutputCol(outputCol)
				val tokenized = tokenizer.transform(inputDataframe)
				(tokenized)
	}
	def removeStopWords(inputDataframe : DataFrame, inputCol : String, outputCol : String,stopwordList : Array[String]) : DataFrame ={
		val remover = new StopWordsRemover().setInputCol(inputCol).setOutputCol(outputCol).setStopWords(stopwordList)
				val cleanedText = remover.transform(inputDataframe)
				(cleanedText)
	}
	def createVector(inputDataframe : DataFrame, inputCol : String, outputCol : String,numFeatures:Int ) : DataFrame ={
		val hashingTF = new  HashingTF().setNumFeatures(numFeatures). setInputCol(inputCol).setOutputCol(outputCol)
				(hashingTF.transform(inputDataframe))     
	}
}