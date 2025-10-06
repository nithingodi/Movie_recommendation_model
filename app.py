import streamlit as st
import boto3
import json
import pandas as pd
import numpy as np
import tmdbsimple as tmdb
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import timm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="AI Movie Concierge", page_icon="🎬", layout="wide")

# --- Load Secrets and Configuration ---
# On Streamlit Community Cloud, you'll set these in the secrets management.
# For local testing in SageMaker, boto3 will use the instance's IAM role automatically.
try:
    SAGEMAKER_ENDPOINT_NAME = st.secrets["aws"]["sagemaker_endpoint_name"]
    AWS_REGION = st.secrets["aws"]["region"]
    tmdb.API_KEY = st.secrets["api_keys"]["tmdb_api_key"]
    # Boto3 will use these credentials when deployed on Streamlit Cloud
    boto3_session = boto3.Session(
        aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"],
        region_name=AWS_REGION
    )
    sagemaker_runtime = boto3_session.client('sagemaker-runtime', region_name=AWS_REGION)
    bedrock_runtime = boto3_session.client('bedrock-runtime', region_name=AWS_REGION)

except (FileNotFoundError, KeyError):
    # This block runs when testing in your SageMaker environment
    # <<< IMPORTANT: PASTE YOUR ENDPOINT NAME AND TMDB KEY FOR LOCAL TESTING >>>
    SAGEMAKER_ENDPOINT_NAME = "endpoint"
    AWS_REGION = "us-west-2" # Your AWS region
    tmdb.API_KEY = "api key"
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)


# --- Caching and Model Loading ---
@st.cache_resource
def load_models():
    """Loads the NLP and CV models into memory."""
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    cv_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    cv_model.eval()
    return nlp_model, cv_model

@st.cache_data
def load_candidate_movies():
    """Fetches a list of candidate movies from TMDb for cold-start recommendations."""
    st.info("Fetching a fresh list of candidate movies from TMDb...")
    discover = tmdb.Discover()
    movies = []
    current_year = pd.to_datetime('now').year
    for i in range(1, 6): # 5 pages = ~100 movies
        movies.extend(discover.movie(page=i, sort_by='popularity.desc', primary_release_year=current_year)['results'])
    return movies

nlp_model, cv_model = load_models()
candidate_movies = load_candidate_movies()

# --- Feature Extraction Functions (for Cold Start) ---
cv_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_data
def get_text_embedding(text):
    return nlp_model.encode(text)

@st.cache_data
def get_visual_embedding(poster_path):
    if not poster_path: return np.zeros(768)
    url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_tensor = cv_transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = cv_model(img_tensor)
        return embedding.cpu().numpy().flatten()
    except:
        return np.zeros(768)

# --- Main App UI ---
st.title("AI Movie Concierge 🎬")

user_type = st.sidebar.radio("Are you a new or returning user?", ["Returning User", "New User"])

if user_type == "Returning User":
    st.sidebar.header("Returning User Login")
    user_ids = list(range(1, 501))
    selected_user_id = st.sidebar.selectbox("Select Your User ID:", options=user_ids, index=74)
    
    if st.sidebar.button("Get My Recommendations"):
        with st.spinner("Calling our AI model on AWS..."):
            input_data = json.dumps({'user_id': selected_user_id})
            response = sagemaker_runtime.invoke_endpoint(
                EndpointName=SAGEMAKER_ENDPOINT_NAME, ContentType='application/json', Body=input_data
            )
            result = json.loads(response['Body'].read().decode())
            
            st.header(f"Recommendations for User {selected_user_id}")
            st.subheader(f"Detected Persona: **{result['persona']}**")
            for rec in result['recommendations']:
                st.markdown(f"#### {rec['title']}")
                st.caption(f"**Reason:** {rec['reason']}")

else: # New User
    st.header("Welcome! Let's find a movie for you.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "What kind of movie are you in the mood for tonight? Feel free to mention a genre, a mood, or a movie you've liked."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("e.g., A thriller like the movie Dark"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting with our AI..."):
                # 1. Use Bedrock to extract structured data
                meta_prompt = f"""You are an AI assistant. The user's request is: '{prompt}'. Extract the genre and a single movie title they mentioned into a JSON object with keys "genre" and "similar_to". Output only the JSON. If a value is missing, use null."""
                body = json.dumps({"inputText": meta_prompt, "textGenerationConfig": {"maxTokenCount": 256, "temperature": 0.1}})
                response = bedrock_runtime.invoke_model(body=body, modelId='amazon.titan-text-express-v1', accept='application/json', contentType='application/json')
                response_body = json.loads(response.get('body').read())
                llm_output = response_body.get('results')[0].get('outputText')
                
                try:
                    extracted_data = json.loads(llm_output)
                    similar_to_title = extracted_data.get("similar_to")
                    
                    if similar_to_title:
                        # 2. Get the "DNA" of the seed movie
                        search = tmdb.Search()
                        response = search.movie(query=similar_to_title)
                        if response['results']:
                            seed_movie = response['results'][0]
                            text_vec = get_text_embedding(f"Title: {seed_movie['title']}. Overview: {seed_movie['overview']}.")
                            vis_vec = get_visual_embedding(seed_movie['poster_path'])
                            temp_user_profile = np.concatenate([text_vec, vis_vec])
                            
                            # 3. Create "DNA" for all candidate movies and find best match
                            candidate_vectors, candidate_titles = [], []
                            for movie in candidate_movies:
                                text_vec = get_text_embedding(f"Title: {movie['title']}. Overview: {movie['overview']}.")
                                vis_vec = get_visual_embedding(movie['poster_path'])
                                candidate_vectors.append(np.concatenate([text_vec, vis_vec]))
                                candidate_titles.append(movie['title'])
                            
                            similarity = cosine_similarity(temp_user_profile.reshape(1, -1), np.array(candidate_vectors))
                            top_indices = similarity.flatten().argsort()[-5:][::-1]
                            
                            st.write("Based on your request, you might like these:")
                            for i in top_indices:
                                st.markdown(f"- **{candidate_titles[i]}**")
                        else:
                            st.write("I couldn't find the movie you mentioned, but here are some popular movies.")
                    else:
                        st.write("I'm sorry, I couldn't identify a specific movie to compare against. Could you try again?")
                except Exception as e:
                    st.error(f"Sorry, I had trouble understanding that. Please try again.")
