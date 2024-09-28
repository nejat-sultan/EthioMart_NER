from telethon.sync import TelegramClient # type: ignore
import os
import pandas as pd # type: ignore


api_id = '27481742'
api_hash = '58e0a00cff4b76e4bd54d6139e0521ea'
phone = '+251900462410'

client = TelegramClient('session_name', api_id, api_hash)

async def get_messages(channel_name, limit=100):
    await client.start(phone)
    channel = await client.get_entity(channel_name)
    
    messages = []
    async for message in client.iter_messages(channel, limit=limit):
        if message.text:  
            messages.append({'sender_id': message.sender_id, 
                             'text': message.text, 
                             'date': message.date})
    
    df = pd.DataFrame(messages)
    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')
    df.to_csv('raw_data/messages.csv', index=False)
    
    return df

with client:
    client.loop.run_until_complete(get_messages('@Leyueqa'))
