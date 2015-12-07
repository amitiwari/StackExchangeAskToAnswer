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
import org.apache.spark.rdd
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
object PredictOne {
  
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
    		val parentId = (xml \ "@ParentId").text
    		val tags = (xml \ "@Tags").text
    		val title = (xml \ "@Title").text
    		val body = (xml \ "@Body").text
    		val bodyPlain = ("<\\S+>".r).replaceAllIn(body, " ")
    		val text = (title + " " + bodyPlain).replaceAll("\n", " ").replaceAll("( )+", " ").replaceAll("\\W", " ");
    		Row(id, postidtype,parentId,tags, text)
		}
		val tagsRDD = tagsXml.map { s =>
    		val xml = XML.loadString(s)
    		val id = (xml \ "@Id").text
    		val tags = (xml \ "@TagName")    
    		val count = (xml \ "@Count")
    		Row(id, tags, count)
		}

		val taglist  = tagsRDD.map { x => x(1).toString()}.collect()
		val postSchemaString = "Id PostTypeId ParentId Tags Text"
		val postSchema = StructType(postSchemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
		val sqlContext = new SQLContext(sc)

		val postsDf = sqlContext.createDataFrame(postsRDD, postSchema).registerTempTable("postsDf")
		val questionsRDD = sqlContext.sql("SELECT * from postsDf where PostTypeId=1")
		val answersRDD = sqlContext.sql("SELECT a.Id, a.PostTypeId, a.ParentId, q.Tags, a.Text from "+
                                    "postsDf as q,postsDf as a where a.PostTypeId=2 AND a.ParentId=q.Id")
                                    
    val tranningRDD = questionsRDD.unionAll(answersRDD);
      
		
	
		val targetTag = "machine-learning"
		val myudf: (String => Double) = (str: String) => {if (str.contains(targetTag)) 1.0 else 0.0}
		val sqlfunc = udf(myudf)
		val postsLabeled = tranningRDD.withColumn("Label", sqlfunc(col("Tags")))
		
		//------------------------------------------split the data in Training and testing set here ------------
		
	
		//Random generate seed
		var r = new scala.util.Random 
		val seed =  r.nextLong()
		val splits__ = postsLabeled.randomSplit(Array(0.7, 0.3), seed = seed)
   
		val training__ = splits__(0)
		val totalTaginTrainset = training__.rdd.filter(line=>line.toString().contains(targetTag)).count()
   
		val test__ = splits__(1)
		val totalTaginTestset = test__.rdd.filter(line=>line.toString().contains(targetTag)).count()
		
		//--------------------------Generate Vectors --------------------------------------------------------------
		
    val training__vector =  new VectorWrapper().vectorize(training__, "Text", stopwordList)
    val  training__vector_labledPoint = training__vector.rdd.map { 
						        row => LabeledPoint(row.getDouble(5), row(8).asInstanceOf[org.apache.spark.mllib.linalg.Vector])
		 }
		
		
		val test__vector =  new VectorWrapper().vectorize(test__, "Text", stopwordList)
    val  test_vector_labledPoint = test__vector.rdd.map { 
						        row => LabeledPoint(row.getDouble(5), row(8).asInstanceOf[org.apache.spark.mllib.linalg.Vector])
		 }
		
		//-------------------------------------Train and Predict--------------------- -----------------
		
		val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training__vector_labledPoint).setThreshold(0.50)
		
		val predictionAndLabels = test_vector_labledPoint.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }
		
		//---------------------------Statistics----------------------------------------
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision
      
      println("Precision = " + precision)
      println("Recall = " +metrics.recall)
      val ct = metrics.confusionMatrix
      println(ct)
      
      var num = ct.apply(0, 0)+ct.apply(1, 1)
      var deno = ct.apply(0, 1)+ct.apply(1, 0)+ct.apply(0, 0)+ct.apply(1, 1)
      
      println("Accuracy " + num/deno)
      
		  val binarymetrics = new BinaryClassificationMetrics(predictionAndLabels)

      val bct = binarymetrics.areaUnderROC()
      println("Area under ROC " +bct)
      
			println("Total tags in testing " + totalTaginTestset)
		  println("Total tags in traning " + totalTaginTrainset)
		
	}

}