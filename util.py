from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import SimpleSequentialChain

template = """
You're an expert in copywriting. You write compelling service descriptions that convert easily. All the user has to do is give you the title of their service, and you're in charge of generating their service description. Use markdown to format your text.
You should provide minimum word count for the service description. For example, if the minimum word count is 500, the service description should be at least 500 words long.
You can use the following variables in your template.
- {serviceTitle}: The service title provided by the user
- {minWord}: The minimum word count provided by the user

You should provide Headings, Sub-headings, bullet point, table, bold text, italic text in the service description 
 
	Service Title:\"\" I'll write a 400-word SEO copy and expert web copy\"\"
	Service Description:\"\"\"
  	Get a Professional Quality SEO ARTICLE to Boost your online PRESENCE!✍
*********************************** NEW 🚀🚀 ******************************
Rank higher in search results with the new GOOGLE TOP POSITION PACK
We use 2023's best On-Page SEO tool to write you a perfectly optimized article that outranks your competitors.  Choose the TOP POSITION GOOGLE pack!  Your article will appear in the Top 3 within a week of going live, no matter how famous your site is!
This article has been optimized with this TOOL https://24-heures-referencement.fr/2023/comment-gagner-de-largent-sur-comeup-ex-5euros-com-en-2023/. Published on February 3, 2023, it is now in the Top 3 of Google and Bing for the query How to make money on comeup in 2023.
We delivered an article optimized with this tool on Feb. 23, 2023. The article is in the Top 7, on the keyword \"change box orange\", one week later.  You can find this article here https://www.mezabo.fr/orange/guide/changer-livebox-ou-decodeur-orange/
******************************************* 🚀🚀 ************************************
Hello :)
You're probably wondering why you should hire Soaredac rather than another seller on Comeup? So here are the good reasons:
✅You'll get high-quality articles written by humans for humans, with up-to-date information, understandable vocabulary, impeccable spelling, faultless grammar, well-developed structure and original, unduplicated content.
✅You'll benefit from the know-how and experience of an experienced SEO Content web copywriter to improve your ranking in search results.
✅ Soaredac's professional SEO copywriter will take care of your blog post or web pages from A to Z, freeing up valuable time to devote to other aspects of your business.
🚀 Over 4,000 customers have been satisfied with the SEO copywriting service provided by Soaredac.
Find out why thousands of customers call on our pen again and again!
You can purchase our basic service for €5 if you want to get an idea of our writing style and quality. This offer includes:
✅ A 400-word blog post optimized for your keyword✅ A catchy title✅ Well-structured paragraphs✅ A text that respects good SEODelivery deadline: 3 days 🕑
Note HOWEVER:
👉🏼 A LONGER article is considered a sign of quality by search engine algorithms, which can lead to better visibility and ranking.
👉🏼Des OPTIMIZED METAS title and description tags have a significant impact on results ranking.
👉🏼 SEMANTIC optimization will enhance the relevance of your content by helping search engines to better understand and evaluate the usefulness of your page for web users.
👉🏼 A Long Article + Semantic Optimization and SEO Optimization with the best ON-Page SEO tool of 2023 will help you rank your site in Google's top position for a specific KEYWORD, just a few days after publication.
Now that you're in the know, dare to take the plunge and find out more about our FORMulas:
Our FORMULAS for all budgets and SEO-optimized EDITORIAL needs!
✍ ACCESS package from €5: applying the basic rules of SEO🔰 GURU package from €10: SEO optimization with the use of Top Terms YourtextGuru🔴 VIP package from €30: optimal semantic optimization with 1.fr🥇 Google Top Position package from €60: optimization with the best On-page SEO tool, based on competitor analyses.
Here are the details:
FORMULA ACCESS and GURU
Copywriting level
✍ ACCESS
🔰 GURU
SEO
Optimization on a main keyword provided by the customer
insertion of Top Terms suggested by YourtextGuru
 
Deliverable (.docx or zip format)
400 words
500-600 words
 
 
Price
5€
10 €
 
Delay for 01 article
3 days
3 days
 
Legibility level
Top
Top
Semantic richness
Average
Good
Keyword density
Average
Good
Optimization Hn tag
-
Good
Optimized writing ux
-
Good
Google Top Position results
-
-
Meta title & description
-
Yes
 
 
Bolding
-
Yes
 

FORMULE PRO
Editing level
🔴 VIP
🥇 Google Top Position Pack
 
SEO
Optimal semantic optimization 70-100% with 1.fr
Optimization with the BEST on-page SEO TOOL in 2023
Deliverable (.docx or zip format)
1000 words
1000 words - 1500 words * Copyscape Verification Capture (anti-plagiarism software) * call to action * 4 royalty-free images (from a free image bank)
Price
30€ (basic service included)
60€
Deadline for 01 article
3 days
5 days
Legibility level
Top
Top
Semantic richness
Top
Top
Keyword density
Top
Top
Optimization Hn tag
High
High
Optimized copywriting ux
Optimal
Optimal
Google Top Position results
A few weeks after publication
About 1 week after publication
Meta title & description
Yes
Yes
 
Bolding
Yes
Yes
 
🛑 IF YOU NEED A CUSTOM OPTION OR ONE NOT ON OUR LIST, DON'T HESITATE TO ASK US AND WE'LL ADD IT BEFORE THE TRANSACTION 🛑
GLOSSARY
Hn tags: titling and subtitling your article
meta Title and meta Description tags: these are the titles and descriptions visible in search engine results
Articles optimized with 1.fr (70 to 100%): semantic optimization with the seo 1.fr copywriting tool is essential for better ranking in Google or Bing results pages (results visible a few weeks after publication of the article).
UX Writing: writing for the user experience
Looking for a VERSATILE SEO EDITOR? Soaredac is MULTI-THEMATIC
OUR THEMES
SUB-THEMES
SEO copywriting for your leisure-related articles
✈ Travel, Sports, Tourism - Shows - Nature and gardening - Art and culture - Outdoor and indoor activities, games and toys
Optimized copywriting for your cooking articles
🍎 Recipes -Food - Dietetics - Nutrition
Web copywriting for your Internet and IT articles
💻 software, applications, digital marketing
SEO copywriting for your fashion articles
💍 Trends - Fashion accessories - Jewelry - Shoes - Young fashion - Vintage
Optimized copywriting for your beauty articles
🕶 Facial care - Hairstyling - Make-up - Nails - Natural products
SEO-optimized copywriting for your articles on the theme of well-being
🦾Meditation - Yoga - Healthy eating - Physical exercise - Lifestyle - Wellness and relaxation center
SEO-optimized copywriting for your home-related articles
🏠Interior decoration - Maintenance - Kitchen and bathroom - Gardening and outdoors - DIY and repairs - Organization and storage
SEO-optimized copywriting for your building-related articles
🏛Construction - Electrical systems - Carpentry - Air conditioning and heating - Plumbing and locksmithing - Interior and exterior insulation - Renovation and restoration
SEO copywriting for your practical life articles
👨👧👧family - lifestyle, shopping - online dating - baby and kids
SEO copywriting for your environmental articles
🍀 Zero waste - ecology - energy saving - renewable energies - conservation and biodiversity
SEO copywriting for your articles on alternative and traditional medicine
🟢 Ayurvedic medicine - Naturopathy - Aromatherapy - Acupuncture - Medicinal plants - Lithotherapy - etc.
Web copywriting for your articles on CBD and Cannabis
🌿 Legality - Forms of consumption - Use - Virtues - Research
Web copywriting for your articles on bodybuilding and fitness
🤺 Exercises - Nutrition and supplements - Training programs - etc.
⚠️Les topics not on our radar :
mechanics (car-bike)
real estate and finance
stock market, trading and forex
pink themes
technical subjects (science, technology, industry, physics, chemistry...)
⛔ Don't hesitate to CONTACT SOAREDAC to make sure your topic falls within its field of expertise before ordering a service. This will save you time.
Want to KNOW MORE about SOAREDAC before you order?
 

🚀 SOAREDAC: 10 YEARS' experience in web copywriting and 5 YEARS' experience on ComeUP
👉 One of the most reliable teams of web copywriters on the platform (high COMPLETION rate)
👉 Very few negative reviews ( LESS than 1%)
👉 Best value in the SEO Redaction category
👉 Competent team: 1 experienced SEO copywriter, 1 versatile web copywriter and 1 meticulous proofreader
👉 Versatile and flexible service: content for web pages or service presentation, corporate seo optimized blog article and netlinking,
👉 Fast and efficient (high DEADLINE Fulfillment Rate).
👉 Web copywriting and seo article writing service, promising results appreciated by Google and readers.
To put your mind at rest, here are a few CUSTOMER REVIEWS!
BL...SE ⭐⭐⭐⭐⭐
I ordered an 1100-word SEO-optimized article. My order was handled with the utmost professionalism (clarity and diligence). I appreciated the finished work for the following reasons: - The semantic richness of the keywords chosen when compared with my list previously built on my side; - The timing: I received the article 1 day in advance; - The originality of the content: I checked the existence or not of plagiarism via two tools including Quetext; - The readability: the text is easy to read with a mix of SEO and attractive content; In conclusion, I will entrust him with the rest of the articles for the same website.
 
 
 
 
 
 
Gui...BC ⭐⭐⭐⭐⭐
Always very satisfied with soaredac's services
Omn...st ⭐⭐⭐⭐⭐
I have just received my article in advance; I have worked with many service providers for the writing of our content and I thank you for the excellent quality presented. I am very pleasantly surprised by the quality of the content! Thank you and see you soon! Best regards.
⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪ ⚡FAQ ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪
 
What is the maximum length of an article we can write?
We can write up to 2500 words per article.
What do we need to write and optimize an article?
- The number of words (for each article)- The theme / title of the article- The main keyword / secondary keywords if any- The desired structure if you have a particular preference (number of subheadings, length of sentences and paragraphs, conclusion...).
What about delivery?
We deliver .docx or .zip files for multiple orders. We undertake to respect the delivery date indicated on the order interface.
Can we deliver as we go along?
We have a number of deadlines to meet every day, so it's difficult for us to deliver as we go.
Do you accept rework requests for every seo optimized text I order?
Yes, and if we have to make a change because of a misunderstanding on our part, we won't charge you for it. On the other hand, ⛔ if we have to make major changes due to a bad brief, you'll have to add a surcharge corresponding to the value of the item.
Can we send you a sample seo copy?
For reasons of confidentiality, we cannot show you sample seo copy written for other clients.
What is seo copywriting?
Seo copywriting is one of the most important aspects of digital marketing. By optimizing your content for search engines, you can attract more traffic and improve your website's visibility. However, SEO copywriting can be tricky. It takes time and practice to master the art. At Soaredac, you're dealing with copywriters who have all undergone seo copywriting training.
 

Why use Soaredac instead of AI?
Although artificial intelligence is becoming an increasingly important part of copywriting, there are still advantages to using a human copywriter. Here are just a few of them:
Creativity: Human copywriters are able to create innovative and engaging content that will capture the attention and interest of readers. AI algorithms can't compete with the creative thinking of a good writer.
Qualitative information: AI cannot provide the same kind of specialized knowledge as a human expert.
Personalization: While robots can produce content quickly, they are not able to tailor that content to your specific target audience or brand with the same finesse as a human. Professional copywriters can incorporate the desired tone and style while providing customized content tailored to your company's specific needs.
SEO optimization: an article produced by AI is semantically poor and requires a significant number of adjustments to achieve a good SEO score. It won't help you achieve high search engine rankings.
Uniqueness: AI-generated texts generally display a high rate of plagiarism. A human copywriter makes sure you get unique work.
Here's a free tool https://copyleaks.com/ai-content-detector that will help you check whether content is written by a human or produced by artificial intelligence.
✍✍Our watchwords?✍✍
 
Rigor
Efficiency
Punctuality
Reliability
 
How to order Soaredac's WEB EDITING service on ComeUp?
Here's a simulation for 2 1200-word ACCESS articles with META TITLE AND META DESCRIPTION
STEP 1 Define your needs1 800-word ACCESS article + meta title and meta description
STEP 2 Customize your order by adding optionsThe basic service already includes an Access article of 400 words max [5€]THEN check the following options:
🅱️ META TITLE AND META DESCRIPTION for 1 to 5 ARTICLES+ 5 €✍️ 1 ARTICLE of 400 words ACCESS formula + 5€=> This service will cost you 15€.
STEP 3 Pay for your purchase (with your 5euros balance, credit card or via PayPal)
STEP 4 Receive your articles
Delivery is in .docx format.
 
OUR OTHER MICROSERVICES on COMEUP
🟥If you'd like us to write guest articles for you, click here Rates from €5
🟥If you'd like product descriptions or product sheets, click here Rates from 5€.
🟥If you'd like copywriting on your Wordpress Blog, click here Rate for 4 publications per month from 50€.
🟥 If you're interested in copywriting for your website, click here] Rates from 5€.
We look forward to working with you.
	\"\"\"
Service Title:\"\" I'll write your 400-word guest article\"\"
Service Description:\"\"\" ⛔The cheapest guest article on the market
▶️ If you're looking for a high-quality guest article writing service? You've come to the right place: SOAREDAC |
||-|| 👉We'll write you a guest article that perfectly matches the editorial charter of your host blog || 👉We've got the best partners to boost your site's SEO || 👉We'll give you quality at the best price! |
For €5, Soaredac will take care of the SEO writing of a guest article of length :
400 words if you add an extra optionOR300 words if you don't add an option
WITHOUT PUBLICATION
What we need to write a quality guest article for you:
The theme of the guest blog article
Is it a sponsored article or not? (Should I praise the destination site? )
Link anchor
URL of destination site
URL of host blog (optional)
Our additional options
* Add royalty-free images
* Add more words for longer articles for €10 to €375
*Editing and publication of a 500-word article on a generalist blog (DA +10 and TF +10) for 15€.
*Writing and publishing a 500-word article on a generalist blog (DA +20 and TF +15) for €20
*Writing and publishing a 500-word article on a generalist blog (DA +50 and TF +30) for 45€.
We can write a paid guest article on a wide variety of topics
marketing guest article
wellness guest article
personal development guest article
health guest article
marketing guest article
lifestyle guest article
fashion guest article
lithotherapy guest article
travel guest article
computer guest article
innovation guest article
and any other type of blog guest article
For writing with publication on one of our general or specialized blogs: The following topics are not accepted: Weapons, Drugs, Medication (Viagra, Cialis), Porn, Sextoys, Cannabis, Hacking, Illegal downloading, Alcohol, Politics, Racism, Religion.
Other microservices offered by Soaredac
🟥If you'd like product descriptions or product sheets, click here Rates from €5
🟥If you'd like to have regular posts published on your Wordpress Blog, click here Rates for 4 posts per month from 50€.
🟥 If you'd like us to edit your website pages, click here] Rates from 5€.
🟥 If you'd like a general or specialized blog post, click here Rate from €5
We look forward to working with you!
	\"\"\"
Service Title :\"\" I'll write a 500-word e-commerce product description\"\"
Service Description :\"\"\"
500 WORD PRODUCT SHEET - Meta title and meta description included - SEO Optimization - E-Commerce Store
Increase your conversions with a high-performance marketing product sheet!
Do you run an e-commerce store? Do you have a great product, but no sales? You've seen on the net that you need to create product sheets if you want to boost your business. So to save money, you decided to write your own sheets. But in the end, you abandoned the initiative. Why did you do this? You realized that you didn't have the time or the skills to do it yourself.
Yet you're aware that without a good product sheet for every item on sale, your online store risks losing potential customers.
Soaredac has the best solution for you! One of our many specialties is the creation of high-converting product sheets. We'll help you create an SEO-optimized marketing product sheet. Our aim is to help you grow your online business and stand out from your competitors.
A clear overview of a product sheet created by Soaredac :
🚀A creative and original online store product sheet totally different from the one provided by the manufacturer.🚀An easy-to-read e commerce product sheet to encourage your visitors to learn more about your offer. 🚀 An informative e-shop product sheet that answers all the questions visitors might have before buying the product🚀 A customized SEO-optimized data sheet for good search engine rankings and to attract potential buyers to you🚀 A product sheet that converts, encouraging visitors to take action and buy your products🚀 A pdf or word product sheet that enhances the user experience on your site.
Who we are
SOAREDAC is a small team of experienced web copywriters. Our ambition is to help you rank higher in the SERPs, by providing you with seo content that's useful for your readers and relevant for Google.
In this microservice, we offer to write a high-quality e-commerce product sheet for your online store. It'll be a promotional item that'll turn your customers' interest into desire, and push them to place an order!
We have the perfect writing style to grab your visitors' attention and push them to make a purchase!  So by adopting our technical data sheet writing microservice, you can be sure of achieving 10x, 100x or 1000x more sales. ~
For €10, we'll produce a seo product sheet with the following features:
- Optimized for Google- Interesting for the reader- 500 words long (meta tags included)- With a catchy title- With title and meta description tags- With a bulleted list- CTA
OR
2 sheets of 250 words with all the content of a 500-word sheet:
Our customers can also purchase several cards in a single order. Here's the list of available options
 
 
 
Designation
Price
Pack of 5 sheets of 500 words OR 10 sheets of 250 words
50€
Pack of 10 sheets of 500 words OR 20 sheets of 250 words
100€
Pack of 20 500-word cards OR 40 250-word cards
200€
Pack of 25 500-word cards OR 50 250-word cards
255€
Pack of 30 500-word cards OR 60 250-word cards
600€
Did you like my microservice sheet? Order yours now to see the results on your sales!
\"\"\"

Service Title to generate :\"\" {serviceTitle}\"\"
Minimum word count :\"\" {minWord}\"\"
Language of the service description :\"\" {language}\"\"
Service Description :\"\"\"

"""

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorDB = Chroma(persist_directory="./vectorDB", embedding_function=embeddings)

# prompt = PromptTemplate(
#    input_variables=["serviceTitle", "minWord", "language"],
#    template=template.strip()
# )


def generate(serviceTitle: str, openai_api_key: str) -> str:
    """
    This function generates the service description based of provided service title
    :param openai_api_key: Your OpenAI API Key
    :param serviceTitle: A string of service title

    :return: Returns a string of service description
    """

    plan_template = """
    You're an expert in copywriting. You write compelling service descriptions that convert easily. All the user has to do is give you the title of their service, and you're in charge of generating their service description. Use markdown to format your text.
    Generate the structure of the service description based on this service title : "{serviceTitle}"
    The structure should contain headings and sub-headings, bullet points, table.
    """

    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)

    # Generate the plan
    plan_prompt_template = PromptTemplate(input_variables=["serviceTitle"], template=plan_template)

    plan_chain = LLMChain(llm=llm, prompt=plan_prompt_template)

    # Generate content for each section of the plan"""

    content_template = """
    You're an web writer. You write compelling service descriptions that convert easily. Write content for each headings and sub-headings, bullet points, table following this structure.
    You should include a table for pricing and delivery time and explain more packages in the service description.
    Structure : {structure}
    """

    llm2 = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=.7, max_tokens=9000, openai_api_key=openai_api_key)

    content_prompt_template = PromptTemplate(input_variables=["structure"], template=content_template)

    content_chain = LLMChain(llm=llm2, prompt=content_prompt_template)

    # Enhance generated service description with Copywriting Skills

    enhancement_template = """
    You're copywriting expert. Use all your knownledge in copywriting fields, all best copywriting techniques to enhance this content.

    Content :
    ------
    {content}
    ------
    """
    enhancement_prompt_template = PromptTemplate(input_variables=["content"], template=enhancement_template)

    enhancement_chain = ConversationalRetrievalChain.from_llm(llm=llm2, retriever=vectorDB.as_retriever(),
                                                              condense_question_prompt=enhancement_prompt_template)

    overall_chain = SimpleSequentialChain(chains=[plan_chain, content_chain, enhancement_chain])

#    llm = ChatOpenAI(
#        model_name="gpt-3.5-turbo-16k",
#        max_tokens=9000,
#        openai_api_key=openai_key
#    )

#    chain = LLMChain(llm=llm, prompt=prompt)

    return overall_chain.run(serviceTitle=serviceTitle)
