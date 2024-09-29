from telethon.sync import TelegramClient  # type: ignore
import os
import pandas as pd  # type: ignore
import asyncio
from dotenv import load_dotenv # type: ignore

load_dotenv()

api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('PHONE')

client = TelegramClient('session_name', api_id, api_hash)

async def get_messages(channel_name, limit=100):
    await client.start(phone)
    channel = await client.get_entity(channel_name)
    
    messages = []

    media_dir = 'media'
    os.makedirs(media_dir, exist_ok=True)

    try:
        async for message in client.iter_messages(channel, limit=limit):
            message_data = {
                'sender_id': message.sender_id,
                'text': message.text if message.text else '',
                'date': message.date,
                'media_path': None
            }

            if message.media and hasattr(message.media, 'photo'):
                filename = f"{channel_name}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)
                message_data['media_path'] = media_path

            messages.append(message_data)

            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    df = pd.DataFrame(messages)

    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')
        
    df.to_csv('raw_data/messages.csv', index=False) 
    
    return df

async def main():
    await get_messages('@Leyueqa')

with client:
    client.loop.run_until_complete(main())
