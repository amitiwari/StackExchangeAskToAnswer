package com.spark.usercommunity
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{BLAS, DenseMatrix, DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.linalg.BLAS

import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.{Logging, SparkContext, SparkException}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.BLAS

object NB {

	def main(args:Array[String]) = {
		val conf = new SparkConf().setAppName(s"Naive Bayes").setMaster("local");
		val sc= new SparkContext(conf)

		//Cache or not ? or count after processing
		val data = sc.textFile("/home/neel/Desktop/test").cache();
		val docCount = data.count();

		val corpus1 = data.map {
			line => val parts = line.split('#')
					val classses = parts(0).split(',')

					val v = for( i<- classses)yield LabeledPoint(i.toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
					(v)
		}.flatMap { line => line }.map(p => (p.label, p.features))
		
		corpus1.foreach(println)
    val createCombiner = (v: Vector) => { (1L, v.copy.toDense)}

		val  mergeValue = (c: (Long, DenseVector), v: Vector) =>  {
		   
		   
		   (c._1 + 1L, c._2)}
		val mergeCombiners = (c1: (Long, DenseVector), c2: (Long, DenseVector)) =>  (c1._1 + c2._1, c1._2)

		val scores = corpus1.combineByKey(createCombiner, mergeValue, mergeCombiners).collect();
		
		scores.foreach(println)
	}
}