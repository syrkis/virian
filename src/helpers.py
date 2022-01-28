# helpers.py
#   virian helper functions
# by: Noah Syrkis

# imports
from boto3.session import Session


def get_s3():
    session = Session()
    client = session.client('s3',
            region_name='AMS3',
            endpoint_url='https://virian.ams3.digitaloceanspaces.com',
            aws_access_key_id='QA3DDQQ6ITF3JMXZOK3H',
            aws_secret_access_key='6Kjt6zx38aOBlOf2HUnxcq9zeA30iVY1Zqs3X3XP03g')
    return client


