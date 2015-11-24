package com.spark.usercommunity
import org.apache.spark.{Logging, SparkContext, SparkException}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.SparkConf
import org.json4s.Xml.{toJson, toXml}
import scala.util.matching.Regex
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SQLContext



object Test {
	def main(args:Array[String]) : Unit= {

			Logger.getLogger("org").setLevel(Level.OFF)
			Logger.getLogger("akka").setLevel(Level.OFF)
			val conf = new SparkConf().setAppName(s"Naive Bayes").setMaster("local[4]");
			val sc= new SparkContext(conf)

			val sqlContext = new SQLContext(sc)
			import sqlContext.implicits._
			
			val  postFile = "/home/neel/datascience.stackexchange.com/Posts.json";

		  val questionsRDD = getQuestionsRDD(postFile, sqlContext)
		  val answersRDD = getAnswersRDD(postFile, sqlContext)
		
	   
		  questionsRDD.registerTempTable("questionsRDD")
		  answersRDD.registerTempTable("answersRDD")
		  
		  //Join questiona nd answer on postid and parentid
		  // Question-id (Parent Id), userWhoHasAnswered, score/upvote/etc, tags
		  val questionAnswerJoind  = sqlContext.sql("SELECT answersRDD._c2, answersRDD._c1, answersRDD._c3, "+
		                                            "questionsRDD._c2 from questionsRDD,answersRDD " + 
	                                             "WHERE questionsRDD._c0 = answersRDD._c2 order by answersRDD._c2")
		  
		  //Explode on tags to get ===>> tag,user,score,question-id
		// val tagsToUserMap = createTagsToUserMaP(questionAnswerJoind.rdd)
		  
		  //tagsToUserMap.foreach { x => println(x) }
		  
		  //tagsToUserMap.registerTempTable("tagsToUserMap")
		  
		 // val groupByTagAndUser = sqlContext.sql("SELECT tagsToUserMap._2, tagsToUserMap._1, SUM(tagsToUserMap._3) " +
		    //                  "FROM  tagsToUserMap GROUP BY tagsToUserMap._2, tagsToUserMap._1 ")
		 
		// println(groupByTagAndUser.take(10))
		 
	}
	
//Selects all the questions with not null tags	
 def getQuestionsRDD(postFile:String,sqlContext :SQLContext)  = {
     val jsonRDD = sqlContext.read.json(postFile)
		 jsonRDD.registerTempTable("jsonRDD")

		val questionsRDD = sqlContext.sql("SELECT row['Id'],row['OwnerUserId'],row['Tags'],"+
		                                     "row['AcceptedAnswerId'],row['Body'],row['AnswerCount']"+ 
		                                    " from jsonRDD WHERE row['Tags'] is not null AND row['PostTypeId']=1")
   	(questionsRDD)
 }
  def getAnswersRDD(postFile:String,sqlContext :SQLContext)  = {
     val jsonRDD = sqlContext.read.json(postFile)
		 jsonRDD.registerTempTable("jsonRDD")

		 val answersRDD = sqlContext.sql("SELECT row['Id'],row['OwnerUserId'],row['ParentId'],"+
		                                        "row['Score'],row['Body'],row['FavoriteCount']"+
		                                        "from jsonRDD WHERE row['PostTypeId']=2 ")
   	(answersRDD)
 }

  def createTagsToUserMaP(questionAnswerJoind : RDD[(String,String,String,String)]) = {/*
    val breakOnTags = questionAnswerJoind.map { x => 
  			val regex = new Regex("""><""")
  			val regex2 = """<|>""".r
  		//	val cleanedImd = regex.replaceAllIn(x(3).toString(), " ")
  			//val cleaned  = regex2.replaceAllIn(cleanedImd, "")
  		//(x.get(0).toString().toInt,x.get(1).toString().toInt,x.get(2).toString().toInt,cleaned.trim().split(" "))
			}.map{x => 
  			val v = for( i<- x._4)yield (i,x._1,x._2,x._3);
  					(v)  
			}.flatMap(x => x)
			(breakOnTags)
  */}
}
