from pyspark import SparkContext
from pyspark import SparkConf

# Setup context
conf = SparkConf().setMaster("local").setAppName("hw3_part2")

# create context
sc = SparkContext(conf=conf)

# data preparation

rawUserArtistData = sc.textFile("s3://aws-logs-523930296417-us-west-2/audio_data/user_artist_data.txt")

userIDStats = rawUserArtistData.map(lambda x: float(x.split(' ')[0])).stats()
itemIDStats = rawUserArtistData.map(lambda x: float(x.split(' ')[1])).stats()
print userIDStats
print itemIDStats

rawArtistData = sc.textFile("s3://aws-logs-523930296417-us-west-2/audio_data/artist_data.txt")

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

rawArtistAlias = sc.textFile("s3://aws-logs-523930296417-us-west-2/audio_data/artist_alias.txt")

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

# data modeling

from pyspark.mllib.recommendation import *


bArtistAlias = sc.broadcast(artistAlias)


def build_rating(x):
    userID, artistID, count = map(lambda elem: int(elem), x.split(' '))
    finalArtistID = bArtistAlias.value.get(artistID)
    if finalArtistID is None:
        finalArtistID = artistID
    return Rating(userID, finalArtistID, count)

trainData = rawUserArtistData.map(build_rating).cache()

model = ALS.trainImplicit(trainData, 10, iterations=5, lambda_=0.01, alpha=1.0)

trainData.unpersist()

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
