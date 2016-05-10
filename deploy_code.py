from pyspark import SparkContext
from pyspark import SparkConf

template_str = 'Step #{num}, {msg}'
base_path = 's3://aws-logs-151918190592-us-east-1/audio_data/'
partition_num = 2

print template_str.format(num=0, msg='Setup the context')

# Setup context
conf = SparkConf().setMaster("local").setAppName("hw3_part2")

# create context
sc = SparkContext(conf=conf)

print template_str.format(num=1, msg='Prepare all the data')

# data preparation

rawUserArtistData = sc.textFile(base_path + "user_artist_data.txt", partition_num)
rawUserArtistData.cache()

userIDStats = rawUserArtistData.map(lambda x: float(x.split(' ')[0])).stats()
itemIDStats = rawUserArtistData.map(lambda x: float(x.split(' ')[1])).stats()
print userIDStats
print itemIDStats

rawArtistData = sc.textFile(base_path + "artist_data.txt")

def artist_parser(elem):
    parts = elem.split('\t')
    if len(parts) != 2:
        return []
    else:
        try:
            return [(int(parts[0]), parts[1])]
        except:
            return []

artistByID = rawArtistData.flatMap(artist_parser)
artistByID.cache()

rawArtistAlias = sc.textFile(base_path + "artist_alias.txt")

def artist_alias_parser(elem):
    parts = elem.split('\t')
    if len(parts) != 2:
        return []
    else:
        try:
            return [(int(parts[0]), int(parts[1]))]
        except:
            return []

artistAlias = rawArtistAlias.flatMap(artist_alias_parser).collectAsMap()

badID, goodID = artistAlias.items()[0]
print str(artistByID.lookup(badID)[0]) + " -> " + str(artistByID.lookup(goodID)[0])

print template_str.format(num=2, msg='build the data for modeling')

# data modeling

from pyspark.mllib.recommendation import *


bArtistAlias = sc.broadcast(artistAlias)


def build_rating(x):
    userID, artistID, count = map(lambda elem: int(elem), x.split(' '))
    finalArtistID = bArtistAlias.value.get(artistID)
    if finalArtistID is None:
        finalArtistID = artistID
    return Rating(userID, finalArtistID, count)

trainData = rawUserArtistData.map(build_rating)
trainData.cache()

print template_str.format(num=3, msg='train the data to get the model')

model = ALS.trainImplicit(trainData, 10, iterations=5, lambda_=0.01, alpha=1.0)

trainData.unpersist()

print template_str.format(num=4, msg='recommend music for user with id 2093760')


userID = 2093760
recommendations = model.call("recommendProducts", userID, 10)

recommendedProductIDs = set(map(lambda r: r.product, recommendations))

rawArtistsForUser = rawUserArtistData.map(lambda r: r.split(' ')) \
                                     .filter(lambda (user, artist, count): int(user) == userID)

existingProducts = set(rawArtistsForUser.map(lambda (user, artist, count): int(artist)).collect())

print "Artists that this user has listened:"
for artist in artistByID.filter(lambda (id, name): id in existingProducts).values().collect():
    print str(artist)    

print "Artists that we recommend this user to listen:"
for artist in artistByID.filter(lambda (id, name): id in recommendedProductIDs).values().collect():
    print str(artist)
