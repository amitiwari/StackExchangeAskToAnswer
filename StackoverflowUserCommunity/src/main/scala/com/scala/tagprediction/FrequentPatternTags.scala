package com.scala.tagprediction
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
object FrequentPatternTags {
      def main(args:Array[String]) = {
        val conf = new SparkConf().setAppName("LG").setMaster("local[3]");
		    val sc= new SparkContext(conf)	
        val data = sc.textFile("/home/neel/Desktop/ff.txt/part-00000")
    
         val transactions: RDD[Array[String]] = data.map(s => s.trim.split(' ')).cache()
    
        val fpg = new FPGrowth()
          .setMinSupport(0.01)
          .setNumPartitions(1)
        val model = fpg.run(transactions)
    
       model.freqItemsets.collect().foreach { itemset =>
          println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
      }
    
 /*   val minConfidence = 0.8
    model.generateAssociationRules(minConfidence).collect().foreach { rule =>
      println(
        rule.antecedent.mkString("[", ",", "]")
          + " => " + rule.consequent .mkString("[", ",", "]")
          + ", " + rule.confidence)
    }*/
      }
}