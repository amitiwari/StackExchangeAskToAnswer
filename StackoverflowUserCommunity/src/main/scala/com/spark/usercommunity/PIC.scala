package com.spark.usercommunity
import org.apache.spark.mllib.clustering.{PowerIterationClustering, PowerIterationClusteringModel}
import org.apache.spark.mllib.linalg.Vectors   
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.io.Source._
import scala.xml._
import scala.math._
import scala.collection.mutable.HashMap
object PIC {
    def main(args:Array[String]) = {
		val conf = new SparkConf().setAppName("CLuster").setMaster("local");
		val sc= new SparkContext(conf)
		val tagfileName = "/home/neel/datascience.stackexchange.com/Tags.xml"
		
		val tagsFile = sc.textFile(tagfileName)
		val tagsXml = tagsFile.map(_.trim).filter(!_.startsWith("﻿<?xml version=")).filter(_ != "<tags>").filter(_ != "</tags>")

		val tagsRDD = tagsXml.map { s =>
    		val xml = XML.loadString(s)
    		val id = (xml \ "@Id").text
    		val tags = (xml \ "@TagName").text    
    		(id, tags)
		}.collect()
		
		var tagToIdMap = new HashMap[Long,String]()
		var IdToTagMap = new HashMap[String,Long]()
    for (i <- tagsRDD) {
       tagToIdMap += i._1.toLong -> i._2
       IdToTagMap += i._2 -> i._1.toLong
    }

		println(IdToTagMap)
		  
    val data = sc.textFile("/home/neel/726/project/pic.txt")
    val similarities = data.map { line =>
    val parts = line.split(',')
        println(parts)
         (parts(0).toLong, IdToTagMap.apply(parts(1)), abs(parts(2).toDouble))
      }
  
  // Cluster the data into two classes using PowerIterationClustering
      val pic = new PowerIterationClustering()
         .setK(3)
          .setMaxIterations(10)
      val model = pic.run(similarities)
   
    val r =  model.assignments.groupBy {x => x.cluster}  
      r.foreach{x => 
        for(i<-x._2){
          if(tagToIdMap.keySet.contains(i.id)){
                 println(x._1 +" "+tagToIdMap.apply(i.id))
          }

        } 
      
      }
   /*  model.assignments.foreach { a =>
      if(tagToIdMap.keySet.contains(a.id)){
         println(s"${tagToIdMap.apply(a.id)} -> ${a.cluster}")
      }  */     
  }
    
}