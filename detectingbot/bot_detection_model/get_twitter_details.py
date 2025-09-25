import csv
from django.conf import settings
import tweepy
import pandas as pd

class TwitterDetails():
    def __init__(self):
        self.base_dir=settings.BASE_DIR
        # self.api_key='ZSuJgFqZnJ1wtt73wgNGIDY2F'
        # self.api_secret_key='tnCtQxxekm62t9nE68vPqu2B3rSBN4WumtfTnJxjbI6jlgG2ZS'
        # self.access_token_key='1300072646986534912-QY90Szvw3TvxbvnFoySvESleHv3UDa'
        # self.access_secret_key='OtRT8m0NDhlzatikJ0RbKDqgy1irzhO5owBDkmpfJeJVR'
        self.bearer_token='AAAAAAAAAAAAAAAAAAAAAOWvaAEAAAAAx5a%2BFBx9Z%2FI1xdDUo2iH%2FnoMUMU%3DVsJxNCVjEX2Opc9ZsFjDmy4OSXLHGW920sd4pEkre0tt0QBzHH'
        #self.client= tweepy.Client(bearer_token=bearer_token, consumer_key=api_key, consumer_secret= api_secret_key, access_token= access_token_key, access_token_secret=access_secret_key,return_type=dict,wait_on_rate_limit=False)
        self.client= tweepy.Client(bearer_token=self.bearer_token,return_type=dict,wait_on_rate_limit=False)

    def get_tweets_details(self, id):
        tweet_fields = ["created_at", "text", "source","lang"]
        user_fields = ["name", "username", "location", "verified", "description"]
        expansions='author_id'
        tweet_details= self.client.get_tweet(id=id,tweet_fields=tweet_fields,user_fields=user_fields,expansions=expansions)
        return None

    def get_user_details(self,username):
        user_fields = ["name", "location", "verified", "description","username","created_at","url","public_metrics"]
        response= self.client.get_user(username=username,user_fields=user_fields)
        field_names=['id','id_str','screen_name','location','description','url','followers_count','friends_count','listed_count','created_at','favourites_count','verified','statuses_count','lang','status','default_profile','default_profile_image','has_extended_profile','name']
        user_details={}
        user_details["id"]=response['data']['id']
        user_details["id_str"]=str(response['data']['id'])
        user_details["screen_name"]=response['data']['username']
        if 'location' in response['data']:
            user_details["location"]=response['data']['location']
        else:
            user_details["location"]=None
        user_details["description"]=response['data']['description']
        user_details["url"]=response['data']['url']   
        user_details["followers_count"]=response['data']['public_metrics']['followers_count']
        user_details["friends_count"]=response['data']['public_metrics']['following_count']
        user_details["listed_count"]=response['data']['public_metrics']['listed_count']
        user_details["created_at"]=response['data']['created_at']
        user_details["favourites_count"]=None
        user_details["verified"]=response['data']['verified']
        user_details["statuses_count"]=response['data']['public_metrics']['tweet_count']
        user_details["lang"]='en'
        user_details["status"]=None
        user_details["default_profile"]=False
        user_details["default_profile_image"]=False
        user_details["has_extended_profile"]=None
        user_details["name"]=response['data']['name']     
        with open(self.base_dir / 'bot_detection_model' / 'datasets' / 'test.csv','w',encoding='utf-8') as datafile:
            writer=csv.DictWriter(datafile,fieldnames=field_names)
            writer.writeheader()
            writer.writerows([user_details])
