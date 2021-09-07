# Virian Pentaglyph Dimensions

*This repository is under active development and things might break*

With the intention of mapping dates to points in Haidt space, document embeddings that can be meaningfully expressed as a linear combination of the Haidt basis vectors must be constructed.
Currently, assuming a samples dimensions to be 512 (word count) x 768 (embedding dimensions), we need to inteligently collapse our samples to be of size 1 x 768.
Thus could be done using a tf-idf weigthed average.
However, a neural approach might be more appropriate.
The priority is to conserve meaning during the transformation process.
Consevation of meaning could, perhaps be achived by having a second loss function during training, auto encoding the word embeddings for the Haidth dimensions.
