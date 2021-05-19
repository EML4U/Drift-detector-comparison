# Example on how to use the model
# Please compute vectors only once, cache and reuse them,
# as models may not generate reproducible results.

import gensim
from gensim.models.doc2vec import Doc2Vec

# https://hobbitdata.informatik.uni-leipzig.de/EML4U/2021-05-17-Amazon-Doc2Vec/
model = Doc2Vec.load("../../../DATA/EML4U/amazon-reviews/amazonreviews_c.model")

# first entry
review_summary = "There Is So Much Darkness Now ~ Come For The Miracle"
review_text = "Synopsis: On the daily trek from Juarez, Mexico to El Paso, Texas an ever increasing number of female workers are found raped and murdered in the surrounding desert. Investigative reporter Karina Danes (Minnie Driver) arrives from Los Angeles to pursue the story and angers both the local police and the factory owners who employee the undocumented aliens with her pointed questions and relentless quest for the truth.<br /><br />Her story goes nationwide when a young girl named Mariela (Ana Claudia Talancon) survives a vicious attack and walks out of the desert crediting the Blessed Virgin for her rescue. Her story is further enhanced when the Wounds of Christ (stigmata) appear in her palms. She also claims to have received a message of hope for the Virgin Mary and soon a fanatical movement forms around her to fight against the evil that holds such a stranglehold on the area.<br /><br />Critique: Possessing a lifelong fascination with such esoteric matters as Catholic mysticism, miracles and the mysterious appearance of the stigmata, I was immediately attracted to the '05 DVD release `Virgin of Juarez'. The film offers a rather unique storyline blending current socio-political concerns, the constant flow of Mexican migrant workers back and forth across the U.S./Mexican border and the traditional Catholic beliefs of the Hispanic population. I must say I was quite surprised by the unexpected route taken by the plot and the means and methods by which the heavenly message unfolds.<br /><br />`Virgin of Juarez' is not a film that you would care to watch over and over again, but it was interesting enough to merit at least one viewing. Minnie Driver delivers a solid performance and Ana Claudia Talancon is perfect as the fragile and innocent visionary Mariela. Also starring Esai Morales and Angus Macfadyen (Braveheart)."
doc = review_summary + " " + review_text

tokens = gensim.utils.simple_preprocess(doc)
print(tokens)
vector = model.infer_vector(tokens)
print(vector)

tokens = gensim.utils.simple_preprocess(doc)
print(tokens)
vector = model.infer_vector(tokens)
print(vector)

# last entry
review_summary = "The Earth is Hollow!"
review_text = "Now for the first time, this film presents the history, mythology & folklore that the earth has a hollow realm, a mystical and physical place, thought to house prehistoric animals, or hide alien beings bent on conquering the earth. The hollow earth theory is represented in the history of many diverse cultures throughout the world. The Avalon of Camelot, the Garden of Eden, Paradise Lost, Shangri-La and Valhalla are names assigned to a mystical and physical place thought by some to house prehistoric animals and plants and by others to hide alien beings bent on conquering the outer Earth. This fascinating video is a compilation of extensive research by the International Society for a the Complete Earth.* DVD * 40 min<br /><br />Rent me: UFOdvd com/rent/"
doc = review_summary + " " + review_text

tokens = gensim.utils.simple_preprocess(doc)
print(tokens)
vector = model.infer_vector(tokens)
print(vector)

tokens = gensim.utils.simple_preprocess(doc)
print(tokens)
vector = model.infer_vector(tokens)
print(vector)