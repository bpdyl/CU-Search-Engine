"""
Training Data for Document Classification

Contains 400+ labeled documents for Business, Entertainment, and Health categories.
Includes long-form articles, medium-length summaries, and short-form queries/phrases
to enable the classifier to handle various text lengths effectively.

Documents are inspired by and representative of content from public news sources
including BBC News, Reuters, and other major news outlets.

Data Sources Attribution:
- Content style based on BBC News (https://www.bbc.com/news)
- Content style based on Reuters (https://www.reuters.com)
- Content style based on The Guardian (https://www.theguardian.com)
- Content style based on CNN (https://www.cnn.com)

Note: These are original compositions inspired by news article styles,
not direct copies of copyrighted content.
"""

def get_training_data():
    """
    Get comprehensive training data for the classifier.

    Returns:
        List of dictionaries with 'text' and 'category' keys
    """

    # ============================================================
    # BUSINESS ARTICLES (50+ documents)
    # ============================================================
    business_articles = [
        # Stock Market & Trading
        "Global stock markets surged today after the Federal Reserve signaled it would pause interest rate hikes, boosting investor confidence across major indices including the S&P 500 and Dow Jones Industrial Average.",
        "Wall Street closed higher on Friday as technology stocks led gains, with the Nasdaq Composite rising 2.3 percent amid optimism about corporate earnings in the tech sector.",
        "Asian markets opened mixed on Monday following concerns about China's economic slowdown and its potential impact on global trade and commodity prices.",
        "European stocks fell sharply after weaker-than-expected manufacturing data from Germany raised concerns about the region's economic health and potential recession.",
        "The FTSE 100 index reached a new all-time high as mining and energy stocks rallied on the back of rising commodity prices and strong quarterly results.",

        # Corporate News
        "Apple announced record quarterly revenue of $123 billion, driven by strong iPhone sales and growing services business, exceeding analyst expectations.",
        "Microsoft completed its acquisition of the gaming company for $69 billion, marking the largest deal in the technology industry's history.",
        "Amazon reported a significant increase in cloud computing revenue as more businesses migrate their operations to AWS infrastructure.",
        "Tesla shares dropped 8 percent after the electric vehicle maker reported lower-than-expected deliveries for the third quarter.",
        "Google's parent company Alphabet announced plans to lay off 12,000 employees as part of a broader cost-cutting initiative across the tech industry.",
        "Meta Platforms invested $10 billion in virtual reality and metaverse development despite reporting declining advertising revenue.",
        "Samsung Electronics reported a 30 percent decline in quarterly profits due to weak demand for memory chips and smartphones.",
        "Netflix added 8 million new subscribers after launching its ad-supported tier, beating Wall Street expectations for the streaming giant.",

        # Banking & Finance
        "JPMorgan Chase reported record annual profits of $48 billion as rising interest rates boosted the bank's lending margins significantly.",
        "The European Central Bank raised interest rates by 50 basis points to combat persistent inflation across the eurozone.",
        "Goldman Sachs announced plans to cut 3,200 jobs as investment banking revenue declined amid a slowdown in mergers and acquisitions.",
        "Citigroup agreed to pay $400 million to settle regulatory charges related to risk management failures in its trading operations.",
        "HSBC reported strong quarterly earnings driven by its Asian operations, particularly in wealth management and commercial banking.",
        "Bank of England held interest rates steady at 5.25 percent while signaling that further increases may be necessary to control inflation.",
        "Credit Suisse was acquired by UBS in an emergency rescue deal worth $3.2 billion after a crisis of confidence in the Swiss bank.",

        # Economy & Trade
        "US unemployment rate fell to 3.4 percent, the lowest level in 54 years, as employers added 517,000 jobs in January.",
        "China's economy grew by 5.2 percent in 2023, meeting the government's target despite challenges from the property sector crisis.",
        "UK inflation dropped to 4 percent in December, down from its peak of 11 percent, raising hopes for interest rate cuts.",
        "The World Bank lowered its global growth forecast to 2.1 percent amid concerns about high interest rates and geopolitical tensions.",
        "India's GDP grew by 7.6 percent, making it the fastest-growing major economy and attracting significant foreign investment.",
        "Japan's central bank surprised markets by adjusting its yield curve control policy, sending the yen sharply higher against the dollar.",
        "Germany officially entered a technical recession after two consecutive quarters of economic contraction amid energy price concerns.",
        "US trade deficit narrowed to $67 billion in November as exports increased and imports of consumer goods declined.",

        # Energy & Commodities
        "Oil prices jumped above $90 per barrel as OPEC+ members announced additional production cuts to support the market.",
        "Natural gas prices in Europe fell to their lowest level in two years as mild winter weather reduced heating demand.",
        "Gold prices reached a record high of $2,100 per ounce as investors sought safe-haven assets amid banking sector turmoil.",
        "Copper prices rallied on expectations of increased demand from China's economic reopening and green energy transition.",
        "BP reported annual profits of $28 billion, the highest in the company's 114-year history, amid elevated oil and gas prices.",
        "Shell announced plans to increase shareholder returns through $4 billion in additional share buybacks over the next quarter.",

        # Retail & Consumer
        "Walmart reported strong holiday sales as consumers sought value amid inflation, with comparable store sales rising 8 percent.",
        "Amazon Prime Day generated $12 billion in sales, setting a new record for the annual shopping event despite economic concerns.",
        "Target shares fell after the retailer warned of declining sales and increased theft affecting its quarterly results.",
        "Starbucks raised prices for the fourth time this year as the coffee chain battles rising labor and ingredient costs.",
        "Nike reported better-than-expected quarterly results as demand for athletic footwear remained strong in North America.",
        "McDonald's raised menu prices by an average of 10 percent to offset higher food and labor costs across its restaurants.",

        # Real Estate & Housing
        "US home sales fell for the twelfth consecutive month as high mortgage rates continued to dampen buyer demand.",
        "Commercial real estate values declined by 15 percent as office vacancy rates reached record highs in major cities.",
        "Housing starts increased unexpectedly in February as builders responded to a shortage of existing homes for sale.",
        "Mortgage rates topped 7 percent for the first time since 2007, pushing homeownership further out of reach for many Americans.",
        "Chinese property developer Evergrande defaulted on its offshore bonds, raising concerns about the country's real estate sector.",

        # Technology Business
        "Artificial intelligence startup OpenAI was valued at $29 billion in a new funding round led by Microsoft and venture capital firms.",
        "Semiconductor stocks surged after Nvidia reported record revenue driven by explosive demand for AI chips and data center products.",
        "IBM announced a major restructuring to focus on cloud computing and artificial intelligence, cutting 3,900 jobs in the process.",
        "Intel shares dropped after the chipmaker reported disappointing results and warned of continued weakness in PC demand.",
        "Salesforce acquired Slack for $27.7 billion to strengthen its position in enterprise collaboration and communication software.",
    ]

    # ============================================================
    # ENTERTAINMENT ARTICLES (50+ documents)
    # ============================================================
    entertainment_articles = [
        # Movies & Box Office
        "The latest Marvel superhero film shattered box office records, earning $450 million globally in its opening weekend and becoming the highest-grossing movie of the year.",
        "Christopher Nolan's new epic war drama received standing ovations at the Cannes Film Festival and is already generating Oscar buzz for its stunning cinematography.",
        "The animated sequel from Pixar delighted audiences worldwide, earning $200 million domestically and receiving praise for its emotional storytelling.",
        "Horror film 'The Conjuring 4' dominated the Halloween box office, scaring up $65 million and cementing the franchise's status as a genre powerhouse.",
        "The biographical drama about the legendary musician earned critical acclaim and is expected to sweep the upcoming awards season.",
        "Director Denis Villeneuve's science fiction epic broke IMAX records, with audiences flocking to see the visually stunning adaptation.",
        "The romantic comedy starring popular actors became an unexpected hit, reviving hopes for the genre in the streaming era.",
        "James Cameron's long-awaited sequel earned $2 billion worldwide, becoming only the third film in history to reach that milestone.",
        "The independent film from a first-time director won the top prize at Sundance Film Festival and was acquired by a major studio.",
        "A24's latest psychological thriller has become a cultural phenomenon, sparking countless discussions and theories on social media.",

        # Television & Streaming
        "HBO's fantasy series finale drew 19 million viewers, making it the most-watched episode in the network's history.",
        "Netflix announced the renewal of its hit Korean drama for a second season after it became the most-watched non-English series on the platform.",
        "The true crime documentary series topped streaming charts for six consecutive weeks, reigniting public interest in the decades-old case.",
        "Disney+ revealed its upcoming slate of Star Wars series, promising five new shows set in the beloved galaxy far, far away.",
        "The comedy series swept the Emmy Awards, winning Outstanding Comedy Series for the third consecutive year.",
        "Apple TV+ scored a major coup by signing a multi-year deal with the acclaimed director for exclusive content.",
        "The reality competition show returned with record ratings as viewers tuned in to watch the dramatic season premiere.",
        "Amazon Prime Video's new action series starring the popular actor became the streaming service's biggest debut ever.",
        "The long-running medical drama announced it will end after 20 seasons, with a special two-hour series finale planned.",
        "BBC's period drama captivated audiences worldwide, with its elaborate costumes and intricate plot becoming topics of viral discussion.",

        # Music
        "Taylor Swift's concert tour became the first to gross over $1 billion, with tickets selling out within minutes across all venues.",
        "The K-pop group broke YouTube records with their new music video, accumulating 100 million views in just 24 hours.",
        "Beyoncé's Renaissance World Tour earned $500 million, cementing her status as one of the highest-grossing touring artists ever.",
        "The rapper's posthumous album debuted at number one on the Billboard charts, with fans celebrating his musical legacy.",
        "Ed Sheeran announced a massive stadium tour for next year, with dates across Europe, North America, and Asia.",
        "The legendary rock band announced their farewell tour after 50 years of performing together, selling out arenas worldwide.",
        "Bad Bunny became the most-streamed artist on Spotify for the third consecutive year, dominating Latin music globally.",
        "The country music star's crossover hit topped both pop and country charts, breaking genre barriers.",
        "Adele's residency in Las Vegas sold out instantly, with tickets reselling for thousands of dollars on secondary markets.",
        "The classical crossover artist achieved viral fame on TikTok, introducing millions of young fans to orchestral music.",

        # Celebrity News
        "Hollywood power couple announced their divorce after 12 years of marriage, shocking fans and dominating entertainment headlines.",
        "The actress revealed her struggles with mental health in a candid interview, sparking important conversations about wellness in the industry.",
        "Celebrity chef opened a new restaurant in London, with reservations booked solid for the next three months.",
        "The royal family released new photographs to celebrate the princess's milestone birthday, delighting fans worldwide.",
        "Talk show host signed a record-breaking contract to continue hosting her popular late-night program for five more years.",
        "The athlete made a surprise cameo in the superhero film, delighting fans with his unexpected acting debut.",
        "Fashion designer launched a sustainable clothing line in partnership with the environmental organization.",
        "The social media influencer's beauty brand was acquired by a major cosmetics company in a deal worth $600 million.",

        # Gaming & Entertainment
        "The highly anticipated video game sequel sold 10 million copies in its first week, breaking the franchise's previous record.",
        "Sony announced the release date for the PlayStation 6, promising revolutionary graphics and immersive gaming experiences.",
        "The esports tournament drew 5 million concurrent viewers, surpassing traditional sports events in online viewership.",
        "Nintendo revealed the successor to the Switch console, generating massive excitement among gaming enthusiasts.",
        "The mobile game phenomenon celebrated its seventh anniversary with special in-game events and record daily active users.",
        "Microsoft's Game Pass subscription service reached 30 million subscribers, validating its strategy of gaming accessibility.",

        # Awards & Events
        "The Academy Awards ceremony featured several historic wins, including the first Asian actress to win Best Actress.",
        "Grammy Awards returned to Los Angeles with spectacular performances and unexpected winners across major categories.",
        "The Tony Awards celebrated Broadway's triumphant return with emotional tributes and record-breaking productions.",
        "Comic-Con International returned to full capacity, with studios unveiling exclusive trailers and star-studded panels.",
        "The Met Gala's avant-garde theme inspired dramatic red carpet looks, with celebrities pushing fashion boundaries.",
    ]

    # ============================================================
    # HEALTH ARTICLES (50+ documents)
    # ============================================================
    health_articles = [
        # Medical Research & Treatments
        "Scientists announced a breakthrough in cancer treatment using mRNA technology, showing promising results in early clinical trials for melanoma patients.",
        "Researchers at Johns Hopkins University developed a new blood test that can detect multiple types of cancer at early stages with 90 percent accuracy.",
        "A revolutionary gene therapy received FDA approval for treating sickle cell disease, offering hope to thousands of patients suffering from the genetic condition.",
        "Clinical trials showed that the new Alzheimer's drug slowed cognitive decline by 27 percent, marking significant progress in treating the devastating disease.",
        "Scientists discovered a new antibiotic compound effective against drug-resistant bacteria, addressing the growing threat of antimicrobial resistance.",
        "Researchers successfully used CRISPR gene editing to treat a patient with a rare genetic disorder, opening new possibilities for genetic medicine.",
        "A new study found that a common diabetes medication may help prevent heart disease in patients without diabetes.",
        "Medical teams achieved a milestone in xenotransplantation by successfully transplanting a genetically modified pig kidney into a human patient.",
        "Scientists identified new biomarkers that could predict Parkinson's disease years before symptoms appear, enabling earlier intervention.",
        "The experimental vaccine for respiratory syncytial virus showed 85 percent efficacy in protecting older adults from severe illness.",

        # Public Health
        "The World Health Organization declared the end of the global health emergency, though cautioning that the virus continues to circulate worldwide.",
        "Health officials reported an increase in measles cases linked to declining vaccination rates in several communities across the country.",
        "A new study revealed that air pollution contributes to 9 million premature deaths globally each year, highlighting urgent environmental health concerns.",
        "The CDC issued new guidelines for preventing the spread of respiratory infections, emphasizing improved ventilation and hand hygiene.",
        "Public health experts warned of a potential influenza surge this winter, urging vulnerable populations to receive their annual flu vaccination.",
        "Global efforts to eradicate polio achieved a major milestone, with cases falling to historic lows across endemic regions.",
        "Health authorities implemented new screening programs for hepatitis C, aiming to eliminate the disease by 2030.",
        "The opioid crisis claimed over 80,000 lives last year, prompting renewed calls for improved addiction treatment and prevention programs.",
        "Water contamination in the affected community led to widespread health concerns and calls for improved infrastructure.",
        "Tuberculosis cases rose globally for the first time in decades, reversing years of progress against the infectious disease.",

        # Mental Health
        "A major study found that cognitive behavioral therapy is as effective as medication for treating moderate depression in adults.",
        "Mental health apps showed promising results in reducing anxiety symptoms among young adults, according to research published in medical journals.",
        "Experts highlighted the growing mental health crisis among teenagers, with anxiety and depression rates reaching record levels.",
        "New research demonstrated the effectiveness of psychedelic-assisted therapy for treatment-resistant depression in controlled clinical settings.",
        "Workplace mental health programs reduced employee stress and improved productivity, according to a comprehensive corporate wellness study.",
        "Schools implemented new mental health curricula to help students develop emotional resilience and coping strategies.",
        "The pandemic's lasting impact on mental health continues, with therapists reporting increased demand for services.",
        "Veterans' mental health services received additional funding to address the high rates of PTSD and suicide among former service members.",

        # Nutrition & Lifestyle
        "Research confirmed that the Mediterranean diet reduces the risk of heart disease and stroke by up to 30 percent in adults.",
        "Scientists found that intermittent fasting may help improve metabolic health and reduce inflammation markers in the body.",
        "A study linked ultra-processed foods to increased risk of obesity, diabetes, and cardiovascular disease.",
        "Nutrition experts recommended increasing fiber intake to improve gut health and reduce the risk of colorectal cancer.",
        "New guidelines emphasized the importance of limiting added sugars to reduce the risk of chronic diseases.",
        "Research showed that regular physical activity can reduce the risk of developing dementia by 30 percent in older adults.",
        "Plant-based diets gained scientific support as studies demonstrated their benefits for heart health and longevity.",
        "Sleep researchers found that consistent sleep schedules are crucial for maintaining cardiovascular health and cognitive function.",

        # Healthcare System
        "Hospital administrators faced staffing shortages as nurses and doctors reported burnout following years of pandemic-related stress.",
        "Telemedicine usage remained elevated post-pandemic, with patients appreciating the convenience of virtual healthcare appointments.",
        "Healthcare costs continued to rise, with the average family spending over $22,000 annually on insurance premiums and out-of-pocket expenses.",
        "Rural hospitals faced closure threats as financial pressures and workforce challenges mounted in underserved communities.",
        "The government announced plans to negotiate drug prices for Medicare, potentially saving billions in healthcare costs.",
        "Artificial intelligence tools helped radiologists detect breast cancer with greater accuracy in screening mammograms.",
        "Electronic health records improved care coordination but raised concerns about data privacy and cybersecurity risks.",
        "Community health centers expanded services to reach uninsured and underserved populations across the country.",

        # Fitness & Wellness
        "Exercise scientists found that even 10 minutes of daily physical activity can significantly improve cardiovascular health.",
        "Yoga and meditation programs showed measurable benefits for reducing blood pressure and managing chronic pain.",
        "Strength training was recommended for older adults to prevent muscle loss and maintain bone density as they age.",
        "Wearable fitness devices helped users increase their daily step counts and maintain healthier activity levels.",
        "High-intensity interval training proved effective for improving fitness in shorter workout sessions than traditional exercise.",
        "Sports medicine specialists emphasized the importance of proper warm-up routines to prevent injuries in athletes.",
        "Rehabilitation programs helped patients recover mobility and independence after hip and knee replacement surgeries.",
        "Physical therapists developed new protocols for treating long-term effects of viral infections on exercise tolerance.",
    ]

    # ============================================================
    # SHORT-FORM BUSINESS SAMPLES (for short query handling)
    # ============================================================
    short_business = [
        # Stock market terms
        "stock market crash",
        "Dow Jones Industrial Average",
        "S&P 500 index",
        "NASDAQ composite",
        "bull market rally",
        "bear market decline",
        "stock trading volume",
        "market capitalization",
        "earnings per share",
        "quarterly earnings report",

        # Company names and business entities
        "Apple Inc revenue",
        "Microsoft Corporation",
        "Amazon quarterly results",
        "Tesla stock price",
        "Google Alphabet earnings",
        "Meta Facebook business",
        "Netflix subscriber growth",
        "JPMorgan Chase bank",
        "Goldman Sachs investment",
        "Berkshire Hathaway",

        # Business concepts
        "merger and acquisition",
        "initial public offering IPO",
        "venture capital funding",
        "private equity investment",
        "hedge fund returns",
        "cryptocurrency bitcoin",
        "blockchain technology business",
        "supply chain disruption",
        "inflation rate increase",
        "interest rate hike",
        "Federal Reserve policy",
        "economic recession fears",
        "GDP growth forecast",
        "unemployment rate",
        "consumer spending",
        "retail sales report",
        "corporate bankruptcy",
        "profit margin decline",
        "revenue growth",
        "dividend payout",

        # Short business phrases
        "Wall Street trading",
        "Fortune 500 company",
        "business news today",
        "financial markets update",
        "corporate earnings",
        "CEO resignation",
        "startup funding round",
        "market volatility",
        "trade deficit",
        "economic outlook",
    ]

    # ============================================================
    # SHORT-FORM ENTERTAINMENT SAMPLES (for short query handling)
    # ============================================================
    short_entertainment = [
        # TV Shows (including Stranger Things!)
        "Stranger Things Netflix",
        "Stranger Things season",
        "Stranger Things cast",
        "Stranger Things Eleven",
        "Stranger Things Upside Down",
        "Game of Thrones HBO",
        "Breaking Bad series",
        "The Crown Netflix",
        "Wednesday Addams show",
        "Squid Game Korean drama",
        "The Mandalorian Disney",
        "House of the Dragon",
        "The Last of Us HBO",
        "Succession finale",
        "Ted Lasso Apple TV",
        "Yellowstone series",
        "The Bear Hulu",
        "White Lotus HBO",
        "Euphoria season",
        "Bridgerton Netflix",

        # Movies
        "Oppenheimer movie",
        "Barbie film 2023",
        "Avatar sequel",
        "Top Gun Maverick",
        "Spider-Man movie",
        "Marvel Avengers",
        "DC Batman film",
        "Star Wars movie",
        "Jurassic Park sequel",
        "Fast and Furious",
        "Mission Impossible",
        "John Wick movie",
        "Dune Part Two",
        "Indiana Jones",
        "Guardians of Galaxy",

        # Music artists and bands
        "Taylor Swift concert",
        "Taylor Swift Eras Tour",
        "Beyonce Renaissance",
        "Drake new album",
        "Bad Bunny music",
        "BTS K-pop band",
        "Blackpink concert",
        "Ed Sheeran tour",
        "Adele Las Vegas",
        "Harry Styles album",
        "The Weeknd concert",
        "Rihanna Super Bowl",
        "Coldplay tour",
        "Billie Eilish",
        "Olivia Rodrigo",

        # Celebrities and entertainment news
        "Oscar nominations",
        "Grammy Awards",
        "Emmy winners",
        "Golden Globe",
        "Cannes Film Festival",
        "Hollywood celebrity",
        "red carpet fashion",
        "box office record",
        "streaming platform",
        "movie premiere",
        "album release",
        "concert tickets",
        "celebrity wedding",
        "entertainment news",
        "TV show cancelled",

        # Gaming
        "PlayStation 5 game",
        "Xbox Series X",
        "Nintendo Switch",
        "Zelda Tears Kingdom",
        "Grand Theft Auto",
        "Call of Duty",
        "Fortnite gaming",
        "Minecraft update",
        "Hogwarts Legacy game",
        "Elden Ring",

        # Short entertainment phrases
        "binge watch series",
        "season finale",
        "movie trailer",
        "celebrity gossip",
        "fan theories",
        "spoiler alert",
        "Netflix original",
        "Disney Plus show",
        "viral video",
        "trending movie",
    ]

    # ============================================================
    # SHORT-FORM HEALTH SAMPLES (for short query handling)
    # ============================================================
    short_health = [
        # Medical conditions
        "cancer treatment options",
        "diabetes management",
        "heart disease prevention",
        "Alzheimer's disease research",
        "Parkinson's symptoms",
        "COVID-19 vaccine",
        "influenza flu shot",
        "stroke warning signs",
        "arthritis pain relief",
        "asthma treatment",
        "depression therapy",
        "anxiety disorder",
        "ADHD medication",
        "autism spectrum",
        "chronic pain management",

        # Medical treatments and procedures
        "chemotherapy side effects",
        "radiation therapy",
        "organ transplant",
        "heart surgery",
        "knee replacement",
        "hip surgery recovery",
        "blood pressure medication",
        "cholesterol lowering drugs",
        "insulin injection",
        "antibiotic resistance",

        # Health and wellness
        "mental health awareness",
        "physical fitness exercise",
        "healthy diet nutrition",
        "weight loss tips",
        "vitamin supplements",
        "sleep quality improvement",
        "stress management",
        "meditation benefits",
        "yoga health benefits",
        "workout routine",

        # Medical research
        "clinical trial results",
        "FDA drug approval",
        "medical breakthrough",
        "gene therapy research",
        "stem cell treatment",
        "mRNA vaccine technology",
        "immunotherapy cancer",
        "CRISPR gene editing",
        "medical research funding",
        "pharmaceutical development",

        # Healthcare system
        "hospital emergency room",
        "healthcare insurance",
        "Medicare Medicaid",
        "doctor appointment",
        "prescription medication",
        "telemedicine virtual visit",
        "nursing shortage",
        "healthcare costs",
        "medical records",
        "patient care",

        # Short health phrases
        "blood test results",
        "health checkup",
        "symptoms diagnosis",
        "side effects warning",
        "treatment options",
        "recovery time",
        "prevention tips",
        "wellness program",
        "nutrition facts",
        "exercise routine",
        "calorie intake",
        "body mass index BMI",
        "heart rate monitor",
        "blood sugar levels",
        "immune system boost",
    ]

    # ============================================================
    # MEDIUM-LENGTH CONTEXTUAL SAMPLES
    # ============================================================
    medium_business = [
        "Tesla CEO Elon Musk announced major changes to the company's electric vehicle production strategy.",
        "The cryptocurrency market experienced significant volatility as Bitcoin dropped below key support levels.",
        "Amazon Web Services reported strong cloud computing growth despite overall tech sector slowdown.",
        "Small businesses struggle with rising costs and supply chain challenges in the current economy.",
        "The Federal Reserve's interest rate decision impacts mortgage rates and consumer borrowing.",
        "Tech layoffs continue as major companies cut workforce to reduce operational costs.",
        "Retail stocks rally ahead of holiday shopping season with optimistic consumer spending forecasts.",
        "Oil prices surge following OPEC production cut announcement affecting energy sector stocks.",
        "Banking sector faces renewed scrutiny after regional bank failures raise systemic concerns.",
        "Corporate mergers and acquisitions activity slows amid economic uncertainty and higher rates.",
    ]

    medium_entertainment = [
        "Stranger Things fans eagerly await the final season of the hit Netflix supernatural horror series.",
        "The Marvel Cinematic Universe expands with new superhero films and Disney Plus series.",
        "Taylor Swift breaks touring records with her record-breaking Eras Tour concert performances.",
        "Korean drama series continue dominating global streaming charts on Netflix and other platforms.",
        "The Academy Awards ceremony features historic wins and memorable performances from top stars.",
        "Video game industry sees record sales as next-generation consoles drive gaming enthusiasm.",
        "Celebrity couple announces engagement sparking social media frenzy among devoted fans.",
        "New streaming service launches with exclusive content from major Hollywood studios.",
        "Music festival season kicks off with headline performances from top artists worldwide.",
        "Animated film from Pixar receives critical acclaim and strong opening weekend box office.",
    ]

    medium_health = [
        "New study reveals Mediterranean diet significantly reduces risk of cardiovascular disease.",
        "Mental health experts warn of rising anxiety and depression rates among young adults.",
        "FDA approves groundbreaking cancer immunotherapy treatment after successful clinical trials.",
        "Researchers develop promising Alzheimer's drug that slows cognitive decline in patients.",
        "Public health officials urge vaccination as flu season approaches with new variants.",
        "Exercise and physical activity shown to improve brain health and prevent dementia.",
        "Healthcare workers report high levels of burnout following years of pandemic stress.",
        "Telemedicine adoption continues growing as patients prefer convenience of virtual appointments.",
        "Nutrition scientists discover link between gut microbiome and mental health conditions.",
        "Sleep deprivation linked to increased risk of chronic diseases including diabetes and obesity.",
    ]

    # Convert to labeled format
    samples = []

    for text in business_articles:
        samples.append({"text": text, "category": "Business"})

    for text in entertainment_articles:
        samples.append({"text": text, "category": "Entertainment"})

    for text in health_articles:
        samples.append({"text": text, "category": "Health"})

    # Add short-form samples
    for text in short_business:
        samples.append({"text": text, "category": "Business"})

    for text in short_entertainment:
        samples.append({"text": text, "category": "Entertainment"})

    for text in short_health:
        samples.append({"text": text, "category": "Health"})

    # Add medium-length samples
    for text in medium_business:
        samples.append({"text": text, "category": "Business"})

    for text in medium_entertainment:
        samples.append({"text": text, "category": "Entertainment"})

    for text in medium_health:
        samples.append({"text": text, "category": "Health"})

    return samples


def get_data_statistics(samples):
    """Get statistics about the training data."""
    stats = {
        "total": len(samples),
        "by_category": {}
    }

    for sample in samples:
        cat = sample["category"]
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

    return stats
