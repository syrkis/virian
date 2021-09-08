# Research Journal

## September 8th 2021
An auto encoder model going from a 768x1 tf-idf weigthed average vector of word based Distilbert embeddings through a five (number of Haidth dimensions) has been trained on 100,000 samples. The compressed 5d embeddings that I wanted to describe relative to a basis of Haidt dimensions seem to all be pointing in very similar directions of R^5 (as does the Haidth basis). This is a problem. I still believe that meaningful embeddings can be constructed, as peoples choice of what to focus on indicates a hirarchy of values, which in turn, should be able to be mapped to the Hadith dimensions. Some signal should be there, though this innitial simplisitic setup does not seem to be able to get at it. Next could be:
- Make much more sofisticated docuemnt embeddings.
- Use word embeddings that'd better map onto the Haidth conseptual space.
- Train compressor on more data.
- Make compressor more sophisticated (instead of going from 768 to 5)

