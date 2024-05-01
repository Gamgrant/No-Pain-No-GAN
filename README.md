
# No-Pain-No-GAN

## Goal of the project

The goal of this project is to utilize generative modeling to automate the creation of album cover art, leveraging the high-dimensional semantic space of audio and metadata (like genre and artist popularity). By incorporating audio cues and album information, the aim is to generate visually appealing and contextually relevant album covers that reflect the essence of the music. This approach is intended to reduce the high costs and labor typically associated with designing album covers, particularly benefiting independent and smaller artists.

## Data Processing
This project involved creating a comprehensive dataset by extracting detailed album information from various sources. The goal was to gather a wide array of data points for each album, ranging from basic metadata to more specific content like song previews and popularity metrics. The data extraction was performed using several web scraping and API tools across multiple platforms, including Wikipedia, Spotify, and YouTube.

### Detailed Steps and Tools Used:

- **Album Information Extraction from Wikipedia (1980-2024):**
  - Utilized different scraping tools due to format variations in older versus newer Wikipedia pages.
  - Tools used: `BeautifulSoup`, `pandas` (`pd.read_html`), `numpy`, `requests`, `urllib`.

- **Data Collection via Spotify Web API:**
  - Extracted `song_id`, `Album_id`, genre (when available), album cover art link, and 30-second song previews (mp3 format).
  - Managed to adhere to the rate limits imposed by the Spotify Web API to ensure efficient data retrieval without interruptions.

- **Popularity Metrics from YouTube:**
  - Initially attempted to use the YouTube API but switched to manual scraping due to severe API limitations on the number of songs processed.
  - Employed Selenium WebDriver for simulating user interactions in a web browser and used multithreading to scrape song popularity metrics across 4 Chrome tabs for a total of 64,000 songs.

- **Data Cleaning and Preparation:**
  - The collected data were exported in both CSV and JSON formats.
  - Conducted extensive data cleaning steps to address missing genres and other data inconsistencies.

- **Generative Model Application:**
  - Utilized the cleaned dataset to train a generative cross-attention-based model for creating album cover art.

### Metadata Analysis
Relevant metadata contained:

- Artist name
- Song name (most popular per album)
- Genres
- Album name
- Top emotions
We used 4899 songs in our project

![genre](img/genre_pie_chart.png)

## Hume AI
We attempted to use Hume AI API for predicting top-k emotions in audio samples with a goal in mind to extract the emotions and fuse it with embedding space for potentially better, more pertinent art covers. However, due to the nature of Hume AI and it's ability to only process the music exerpts that contain lyrics, we weren't able to extract top-k emotions since most of the extracted .mp3 song previes did not contain the lyrics.
