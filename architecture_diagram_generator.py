import google.generativeai as genai
import os
from typing import List, Tuple, Union
import streamlit_mermaid as stmd
import streamlit as st
import vertexai
from datetime import datetime
import base64
import re
from google.cloud import storage
import string
import random
import tempfile

#Fill in these details for the API_KEY, Project name and the region
API_KEY = "<Your Key>"
vertexai.init(project="<Your Project Name>", location="<Your region>")

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
selected_model = GenerativeModel("gemini-1.5-pro-001")
image_model = genai.GenerativeModel('gemini-1.5-pro-001')


def f_createrandomfilename(vFileName):
    """Cleans a filename for compatibility across operating systems.
    Removes spaces, special characters, and limits length to 255 characters.
    Args:
        filename (str): The filename to clean.
    Returns:
        str: The cleaned filename.
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    vFileName = "".join(c for c in vFileName if c in valid_chars)
    vFileName = vFileName.replace(" ", "_")  # Replace spaces with underscores
    vFileName = vFileName[:255]  # Limit length to 255 characters
    vFileName = "iamanu_" + str(random.randint(1, 1000000)) + "_" + vFileName
    print (vFileName)
    return vFileName

def remove_whitespace_between_alphanumeric(text):
  """
  Removes whitespace between alphanumeric characters in a multiline string
  Args:
    text: The multiline string to process.
  Returns:
    The processed string with whitespace removed between alphanumeric characters.
  """
  processed_lines = []
  for line in text.splitlines():
    if re.search(r'[^a-zA-Z0-9\s&]', line):  # Matches any character that's not alphanumeric or whitespace
         processed_line = re.sub(r'([\w])\s+([\w])', r'\1\2', line)
    else:
         processed_line = line
    processed_lines.append(processed_line)
  return '\n'.join(processed_lines)


st.set_page_config(
    page_title="Architecture Diagram Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)


def f_get_the_local_file_path(vUploadedFile): 
    try:
        vFileName = f_createrandomfilename(vUploadedFile.name)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            vUploadedFile.seek(0)
            temp_file.write(vUploadedFile.read())
            print(temp_file.name)
            return(temp_file.name)
    except Exception as e:
            print(e)
            return(e)


def get_storage_url(gcs_uri: str) -> str:
    """Convert a GCS URI to a storage URL."""
    return "https://storage.googleapis.com/" + gcs_uri.split("gs://")[1]


def get_gemini_response(
    model: GenerativeModel,
    contents: Union[str, List],
    generation_config: GenerationConfig = GenerationConfig(
        temperature=0.1, max_output_tokens=2048
    ),
    stream: bool = True,
) -> str:
    """Generate a response from the Gemini model."""
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    if not stream:
        return responses.text

    final_response = []
    for r in responses:
        try:
            final_response.append(r.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)


def f_curate_response_for_mermaid(response_text):
    try:
                            txt=response_text
                            x = txt.index("```")
                            y= txt[x+3:]
                            z=y.index("```")
                            sss= y[:z]
                            errorfreeresponse = "```"+sss+"```"
                            txt = errorfreeresponse.replace(" {", "{").replace(" }", "}").replace(" ]", "]").replace(" [", "[").replace(" )", ")").replace(" (", "(").replace("style  ", "style ").replace("  px", " px").replace("sub graph","subgraph").replace("subgraph  \"", "subgraph \"")
                            brackets = [")]", ")}", "))"]
                            for pattern in brackets:
                                try:
                                    x = txt.index(pattern)
                                    y= txt.rfind("(",0,x)
                                    z= txt[y+1:x]
                                    listed_string = list(txt)
                                    listed_string[x] = ""
                                    listed_string[y] = ":"
                                    txt=("".join(listed_string))
                                except:
                                    break

                            replaced = txt.replace("mermaid", "").replace("```","")
                            processed_text = remove_whitespace_between_alphanumeric(replaced)
                            replaced = remove_whitespace_between_alphanumeric(processed_text)
                            return replaced
    except Exception as e:
            return(e)


with open("context.txt", "r") as f:
    context = f.read()

# Format the context and prompt   {mermaid_context}
textprompt=f"""
Context:
{context}
You are an experienced Google Cloud Architect who specializes in creating architectural diagrams using Mermaid diagramming tool.
Focus on the technical specifications section in the document and create a Google Cloud Architecture Diagram using Mermaid.js tool. 
Do not use classDef, Style, CSS tags in output.
Enclose the diagram inside a box and label it GCP and under the box show the networking section and for each google cloud product enclose it in a box which mentions its category. 
Eg: Compute Engine maps to  VM, MySQL and CloudSpanner map to Database. Use your knowledge to segment the services in appropriate categories.
The connection order should be Network, Components, Databases.
print(prompt)
"""
imageprompt="I am a solutions Architect and I want you to comprehend the attached image. I want to draw an architecture diagram using GCP services. Help me draw one using mermaid.js code that can be rendered. Come up with an architecture diagram using all of the services mentioned in this image. Do not use CSS or style. Wherever possible and correct,  Put the services inside box which shows the category of services. e.g. Cloud Spanner as Database, BigQuery for Analytics, Looker as Visualization and Cloud Functions for serverless compute."
videoprompt="I am a solutions Architect and I want you to comprehend the attached video. By analyzing the components from this video which can potentially send data, I want to draw an architecture diagram using GCP services in its appropriate category for data processing.Help me draw one using mermaid.js code that can be rendered. Do not apply CSS or Style."


st.header("Architecture Diagram Generation - Made Easy!", divider="rainbow")
html_code = """
   Here we provide you with three different ways to :orange[**generate an architecture diagram**]!\n
   **Generate**: Provides you a reference architecture based upon the services identified in given context.
   **Upload**: Helps you upload an image which can even be hand-drawn. We make it neat using Gemini.
   **Capture**: Literally captures the diagram you drew on a piece of paper or a whiteboard.
   Try it out! 
"""
st.write(html_code)
tab1, tab2, tab3, tab4 = st.tabs(["Generate", "Upload", "Capture", "Video Analysis"])


with tab1:
                generate_arch = st.button("Generate Architecture", key="generate_arch")
                if generate_arch and textprompt:
                    with st.spinner(
                        f":rainbow[Generating Architecture Diagram using given context...]"
                    ):
                        try:
                            response = get_gemini_response(selected_model, textprompt, GenerationConfig(temperature=0, max_output_tokens=2048))
                            replaced = f_curate_response_for_mermaid(response)
                            try:
                                    stmd.st_mermaid(replaced, height= "300px", width= "850px")                                 
                            except:
                                    st.write("Reupload the document or regenerate the architecture.")
                        except Exception as e:
                            print(e)
                            msg ="""
                            graph LR
                                A[Context unclear] --> B(Need more information)
                            """
                            stmd.st_mermaid(msg, height= "350px", width= "auto")

with tab2:
                st.write("""
                     Upload a JPG, JPEG, or PNG image of your rough architecture and let us give you a neater version of it to embed in a PPT""")
                
                vFileExplainChosen = False
                vUploadedFile = st.file_uploader(
                    "", 
                    accept_multiple_files=False, 
                    key="vUploadedFile",
                    type=["jpg", "jpeg", "png"])

                upload_arch = st.button("Generate from Image", key="upload_arch")
                if vUploadedFile is not None:  # Check if a file has been uploaded
                    file_size = len(vUploadedFile.getvalue())  # Now it's safe to call getvalue()
                    if file_size > 10 * 1024 * 1024:  # Check file size
                        st.error("File size exceeds 10 MB limit. Please upload a smaller file.")
                    else:
                        if vUploadedFile.type in ['image/png', 'image/jpeg', 'image/jpg'] and upload_arch:
                            with st.spinner(
                                f":rainbow[Generating Architecture Diagram using the uploaded image ...]"
                            ):
                                try:
                                    vFileContent = base64.b64encode(vUploadedFile.read()).decode('utf-8')
                                    image_display = f'<img src="data:{vUploadedFile.type};base64,{vFileContent}"  width="800" height="350">'
                                    st.markdown(image_display, unsafe_allow_html=True)
                                    vFileExplainChosen = True
                                    file_path = f_get_the_local_file_path(vUploadedFile)
                                    genai.configure(api_key=API_KEY)
                                    sample_file = genai.upload_file(path=file_path, display_name="My Arch PNG", mime_type="image/jpeg")
                                    response = image_model.generate_content([sample_file, imageprompt])
                                    replaced = f_curate_response_for_mermaid(response.text)
                                    stmd.st_mermaid(replaced, height= "550px", width= "auto")
                                except Exception as e:
                                    print(e)
                                    msg ="""
                                    graph LR
                                        A[Context unclear] --> B(Need more information)
                                    """
                                    stmd.st_mermaid(msg, height= "350px", width= "auto")
                        else:
                            st.info("Click the button to start generating the architecture.")

with tab3:      
                vUploadedFile = st.camera_input("Please take a picture of the architecture diagram and upload it in the next step.")
                if vUploadedFile is not None:
                    st.success("Thank you, picture is successfully taken. Please continue to analyze the image now")
                    vFileContent = base64.b64encode(vUploadedFile.read()).decode('utf-8')
                    image_display = f'<img src="data:{vUploadedFile.type};base64,{vFileContent}" width="600" height="350">'
                    st.markdown(image_display, unsafe_allow_html=True)
                    st.write("Name of the image file :", vUploadedFile.name)
                vButtonPicture = st.button("Analyze the diagram", type="primary", key="vButtonPicture")
                if vButtonPicture and vUploadedFile.type in ['image/png', 'image/jpeg', 'image/jpg']:
                    with st.spinner(
                        f":rainbow[Generating Architecture Diagram Based Upon the Captured Screenshot ...]"
                    ):
                        try:
                            vFileExplainChosen = True
                            file_path = f_get_the_local_file_path(vUploadedFile)
                            genai.configure(api_key=API_KEY)
                            sample_file = genai.upload_file(path=file_path, display_name="My Arch PNG", mime_type="image/jpeg")
                            response = image_model.generate_content([sample_file, imageprompt])
                            replaced = f_curate_response_for_mermaid(response.text)
                            stmd.st_mermaid(replaced, height= "1250px", width= "850px")                          
                        except Exception as e:
                                print(e)
                                msg ="""
                                graph LR
                                    A[Context unclear] --> B(Need more information)
                                """
                                stmd.st_mermaid(msg, height= "350px", width= "auto")

with tab4:      
                video_geolocation_uri = ("gs://github-repo/img/gemini/multimodality_usecases_overview/bus.mp4")
                video_geolocation_url = get_storage_url(video_geolocation_uri)
                if video_geolocation_url:
                    video_geolocation_img = Part.from_uri(video_geolocation_uri, mime_type="video/mp4")
                col1,col2=st.columns([2,2],vertical_alignment="center")
                with col1:
                    st.video(video_geolocation_url)
                with col2:
                    analyzevideo = st.button("Analyze this video", key="analyzevideo")
                if analyzevideo:
                    with st.spinner(
                        f":rainbow[Generating Architecture Diagram for the given video...]"
                    ):
                        try:
                            response = get_gemini_response(
                                selected_model, [videoprompt, video_geolocation_img]
                            )
                            replaced = f_curate_response_for_mermaid(response)
                            stmd.st_mermaid(replaced, height= "1250px", width= "850px")                            
                        except Exception as e:
                            print(e)
                            msg ="""
                            graph LR
                                A[Context unclear] --> B(Need more information)
                            """
                            stmd.st_mermaid(msg, height= "550px", width= "auto")
