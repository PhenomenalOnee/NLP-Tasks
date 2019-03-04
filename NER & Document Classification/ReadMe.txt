'Following are two programs regarding NLP TAsk'

'The First Program file:- "Doc_classify" performs document clasification into two cateogories 1)Aggreements 2)Amendments
	 	 	  The aproach here is that i first divide the document in 300 words paragraphs than for each paragragh
	 	 	   we have one label that belongs to that class.
	 	 	  Than i train the data using LSTM RNN network.
	 	 	 I have provided the output file for this program as "Output(Classify)" and have got 96% acuracy.


The Second Program file:- "Phrases_Extract" performs noun phrases and numeric phrases extraction from documents.
	 	 	  Here i uses nltk pos_tag class that tags each word to its POS(Part OF Speech).
 	 	 	  than I perform chunking which extracts phrases based on some grammer syntax.
 	 	 	 I can get better phrases by adjusting the grammer syntax used in extraction but time is not my friend.
 	 	 	 I also want to use Deep Networks in this task but cant do it right now .
 	 	 	 I ahve provided the "output(phrases)" file which contains output of the program.
 	 	 	It shoes top 10 noun phrases extrcted from each documnet and numeric phrases.
	 	 	 Two csv files contains noun_phrses for each documnet(row) and numeric data.	