package com.spark.usercommunity
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
import org.apache.spark.rdd.RDD
object UserTagScore {

  def main(args:Array[String]) : Unit= {
	
    val conf = new SparkConf().setAppName("LG").setMaster("local");
		val sc= new SparkContext(conf)		
		Logger.getLogger("org").setLevel(Level.OFF)
		Logger.getLogger("akka").setLevel(Level.OFF)
    val modelDirectory = "/home/neel/lrmodels"
    val testfileName = "/home/neel/datascience.stackexchange.com/Posts.xml"
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
          		val regex = new Regex("""><""")
  			      val regex2 = """<|>""".r
  			      val cleanedImd = regex.replaceAllIn(tags, " ")
  		        val cleanedTag  = regex2.replaceAllIn(cleanedImd, "")
          		val ParentId = (xml \ "@ParentId").text
          		val Score = (xml \ "@Score").text
          		val OwnerUserId = (xml \ "@OwnerUserId").text
          
          		Row(id, postidtype,cleanedTag,ParentId,Score,OwnerUserId)
  		}
		
		val postsarray = postsRDD.collect();
		val postSchemaString = "Id PostTypeId Tags ParentId Score OwnerUserId"
		val postSchema = StructType(postSchemaString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
		val sqlContext = new SQLContext(sc)
	
		val postsDf = sqlContext.createDataFrame(postsRDD, postSchema).registerTempTable("postsDf")		
		val questionAnswerJoind  = sqlContext.sql("SELECT t2.Id, t2.OwnerUserId, t2.Score, "+
		                                            "t1.Tags from postsDf as t1,postsDf as t2 " + 
	                                            "WHERE t1.Id = t2.ParentId order by t2.ParentId")        
	         
	  //------------------------------------create user,tag, score ---------------------------------------                                          
		val userTagRelationString = "UserId TagId Score"
		val userTagRelationSchema = StructType(userTagRelationString.split(" ").map(fieldName => StructField(fieldName, StringType, true)))
		
		val userTagRelationRDD = questionAnswerJoind.rdd.map { x => 
		    
		  	val taglist= x(3).toString().split(" ")
		  	for(tag <- taglist)yield (Row(x(1),tag,x(2)))
		}.flatMap { x => x }
		
	  val userTagRelationDF = sqlContext.createDataFrame(userTagRelationRDD, userTagRelationSchema).registerTempTable("userTagRelationTable")		  
	  val re = sqlContext.sql("select UserId, TagId, SUM(Score) as TotalScore from userTagRelationTable Group by UserId,TagId ORDER by UserId")
	  re.rdd.coalesce(1).saveAsTextFile("/home/neel/Desktop/graph.txt")
	  
	  re.foreach { println }
		
	}
}
