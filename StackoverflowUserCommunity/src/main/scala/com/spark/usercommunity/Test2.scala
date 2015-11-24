package com.spark.usercommunity
import scala.util.matching.Regex

object Test2 {
  
  def main(args : Array[String]){ 
      val regex = new Regex("""><""")
      val regex2 = """<|>""".r
      val str = "<education><open-source><education>"
      
      val cleaned = regex.replaceAllIn(str, " ")
      val cleaned2 = regex2.replaceAllIn(cleaned, "")
        
      println(cleaned2)
      println(cleaned2.trim().split(" ").length)
     
   }

}