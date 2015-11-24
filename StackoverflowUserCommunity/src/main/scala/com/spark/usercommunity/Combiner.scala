package com.spark.usercommunity

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object Combiner {
	def main(args:Array[String]) = {
		val conf = new SparkConf().setAppName(s"Naive Bayes").setMaster("local");
		val sc= new SparkContext(conf)

		type ScoreCollector = (Int, Double)
		type PersonScores = (String, (Int, Double))

		val initialScores = Array(("Fred", 88.0), ("Fred", 95.0), ("Fred", 91.0), ("Wilma", 93.0), ("Wilma", 95.0), ("Wilma", 98.0))

		val wilmaAndFredScores = sc.parallelize(initialScores).cache()

		val createScoreCombiner = (score: Double) => (1, score)

		val scoreCombiner = (collector: ScoreCollector, score: Double) => {
			val (numberScores, totalScore) = collector
					(numberScores + 1, totalScore + score)
		}

		val scoreMerger = (collector1: ScoreCollector, collector2: ScoreCollector) => {
			val (numScores1, totalScore1) = collector1
					val (numScores2, totalScore2) = collector2
					(numScores1 + numScores2, totalScore1 + totalScore2)
		}
		val scores = wilmaAndFredScores.combineByKey(createScoreCombiner, scoreCombiner, scoreMerger)

				val averagingFunction = (personScore: PersonScores) => {
					val (name, (numberScores, totalScore)) = personScore
							(name, totalScore / numberScores)
				}

				val averageScores = scores.collectAsMap().map(averagingFunction)

						println("Average Scores using CombingByKey")
						averageScores.foreach((ps) => {
							val(name,average) = ps
									println(name+ "'s average score : " + average)
						})

	}
}