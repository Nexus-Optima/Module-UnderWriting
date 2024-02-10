#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pdf2image import convert_from_path


# In[2]:


# Provide the path to the PDF file you want to convert
pdf_path = "10000923_BANK_STMT_1.pdf"

# Enclose the file path in double quotes to handle spaces
images = convert_from_path(pdf_path)


# In[3]:


for i in range(len(images)):
    images[i].save(pdf_path + ' -page' + str(i)+ '.png', 'PNG')


# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract


# In[5]:


def detect_table(file):
    im1 = cv2.imread(file, 0)
    im = cv2.imread(file)
    # invert image and convert to binary
    ret,thresh_value = cv2.threshold(im1,90,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((15,15),np.uint8)
    # dilate text into blobs to get contours
    dilated_value = cv2.dilate(thresh_value,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cordinates = []
    # find coordinates of all contours
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cordinates.append((x,y,w,h))
        # find the biggest contour
        if (y< 10000 and w>250 and h>250):
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)
            roi = im[y:y+h, x:x+w]
    image_height = roi.shape[0]
    padding = int(image_height * 0.1)
    intermediate_img = cv2.copyMakeBorder(roi,padding,padding,padding,padding,cv2.BORDER_CONSTANT, value = [255,255,255])
    cv2.imwrite('output.jpg',intermediate_img)
    return intermediate_img

def convert_image_to_table(intermediate_file):
    modified_im = cv2.imread(intermediate_file)
    modified_im1 = cv2.imread(intermediate_file,0)
    ret,thresh_value = cv2.threshold(modified_im1,128,255,cv2.THRESH_BINARY_INV)
    # identify vertical lines
    vertical_kernel = np.array([[1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1]])
    eroded_image = cv2.erode(thresh_value, vertical_kernel, iterations=5)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
    # identify horizontal lines
    hor_kernel = np.array([[1,1,1,1,1,1]])
    image_2 = cv2.erode(thresh_value, hor_kernel, iterations=5)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=15)
    # identify table structure
    vertical_horizontal_lines = cv2.add(vertical_lines, horizontal_lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated_vertical_horizontal = cv2.dilate(vertical_horizontal_lines,kernel,iterations = 5)
    # identify contours in table structure
    contours, hierarchy = cv2.findContours(dilated_vertical_horizontal,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    widths = [cv2.boundingRect(contour)[2] for contour in contours]
    average_height = np.mean(heights)
    average_width = np.mean(widths)
    counter = 0
    cordinates = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # identify cells of the tabular structure
        if (y< 10000 and (w<3*average_width and h<4*average_height and w>average_width/4 and h>average_height/4)):
            cv2.rectangle(modified_im,(x-2,y-2),(x+w+1,y+h+1),(0,0,255),1)
            cordinates.append((x,y,w,h))
            counter+=1
    cv2.imwrite('extractedtext.jpg',modified_im)
    # create image without table
    final_text_image = cv2.subtract(thresh_value,dilated_vertical_horizontal)
    # sort cells by their y co-ordinate
    cordinates = sorted(cordinates,key = lambda x:x[1])
    table_rows = []
    half_of_mean_height = average_height/3
    current_row = [cordinates[0]]
    # identify cells within the same row by comparing y co-ordinate
    for cnt in cordinates[1:]:
        current_y_coordinate = cnt[1]
        previous_y_coordinate = current_row[-1][1]
        distance_between_y = abs(current_y_coordinate - previous_y_coordinate)
        if distance_between_y <= half_of_mean_height:
            current_row.append(cnt)
        else:
            table_rows.append(current_row)
            current_row = [cnt]
    table_rows.append(current_row)
    # sort cells within rows using their x co-ordinate
    for row in table_rows:
        row.sort(key = lambda x:x[0])
    image_number = 0
    current_row = []
    table = []
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
    # create cropped images of cells and extracting text and storing in table using pytesseract
    for row in table_rows:
        for cnt in row:
            x,y,w,h = cnt
            cropped_image = modified_im[y:y+h, x:x+w]
            image_slice_path = "./OCR/img_"+str(image_number) + ".jpg"
            cv2.imwrite(image_slice_path, cropped_image)    
            results_from_ocr = pytesseract.image_to_string(image_slice_path, lang='eng').strip()
            results_from_ocr = results_from_ocr.replace("\n", " ")
            results_from_ocr = results_from_ocr.replace(",", "")
            current_row.append(results_from_ocr)
            image_number += 1
        table.append(current_row)
        current_row = []
    header = table[0]
    return table, header
    


# In[6]:


final_table = []
initial_header = []
current_header = []
for i in range(len(images)):
    file = pdf_path + ' -page' + str(i)+ '.png'    
    intermediate_img = detect_table(file)
    cv2.imwrite('output' + ' -page' + str(i)+ '.jpg',intermediate_img)
    intermediate_file = 'output' + ' -page' + str(i)+ '.jpg'
    converted_table, header = convert_image_to_table(intermediate_file)
    print(header)
    if i==0:
        initial_header = header
    else:
        current_header = header
    if current_header == initial_header:
        final_table.append(converted_table[1:])
    else:
        final_table.append(converted_table)
    
print(final_table)


# In[7]:


with open("final_output.csv", "w") as f:
    for converted_table in final_table:
        for row in converted_table:
            f.write(",".join(row) + "\n")


# In[8]:


import pandas as pd
from datetime import datetime
import dateutil.parser
import numpy as np
import re


# In[9]:


def convert_str_to_date(date_list):
    final_output=[]
    for i in date_list:
        try:
            final_output.append(dateutil.parser.parse(i.replace("!","")).strftime("%d/%m/%Y"))
        except:
            final_output.append(np.nan)
    return final_output


# In[10]:


# Function to clean a string by keeping only numeric and decimal point characters
def clean_string(s):
    try:
        return re.sub(r'[^0-9.]', '', s)
    except TypeError:
        return s
    
def clean_and_convert(value):
    if pd.notna(value):
        # Check if the value is already numeric
        if isinstance(value, (int, float)):
            return value
        else:
            # Remove non-numeric characters and convert to float
            cleaned_value = ''.join(filter(lambda x: x.isdigit() or x == '.', str(value)))
            try:
                return float(cleaned_value)
            except ValueError:
                return 0
    else:
        return 0

df = pd.read_csv("final_output.csv")
df = df.rename(columns = {'Txn Date':'Date'})
txn_date = convert_str_to_date(df['Date'].tolist())
df['Date'] = txn_date
df = df.dropna(subset = ['Date'])
print(df)
df['Balance'] = df['Balance'].apply(clean_and_convert)
df['Credit'] = df['Credit'].apply(clean_and_convert)
df['Debit'] = df['Debit'].apply(clean_and_convert)
print(df)


# In[11]:


df_abb = df[['Date','Balance']].copy()
df_abb['Date'] = pd.to_datetime(df_abb['Date'], format = "%d/%m/%Y")
if df_abb['Date'].is_monotonic_decreasing:
    # Dates are in descending order, keep the first duplicate
    df_abb = df_abb.drop_duplicates(subset='Date', keep='first')
else:
    # Dates are in ascending order, keep the last duplicate
    df_abb = df_abb.drop_duplicates(subset='Date', keep='last')
end_date = df_abb['Date'].max()
    # Reindex the DataFrame with the complete date range
df_abb = df_abb.set_index('Date')
df_abb = df_abb.asfreq('D')
# Fill in missing values (e.g., using forward fill)
df_abb['Balance'] = df_abb['Balance'].fillna(method = 'ffill')
# Reset the index to have 'Date' as a column again
df_abb = df_abb.reset_index()
six_months_date = end_date.replace(day=1) - pd.DateOffset(months=6)
df_abb=df_abb[df_abb['Date'] >= six_months_date]
monthly_stats = df_abb.groupby(df_abb['Date'].dt.to_period("M")).agg({'Balance': ['sum', 'count']}).reset_index()
print(monthly_stats)
print(df_abb)


# In[12]:


df_transaction = df[['Date','Details','Ref No./Cheque No','Debit','Credit']].copy()
df_creditors = df_transaction[df_transaction['Credit'] != 0].copy()
df_creditors['Date'] = pd.to_datetime(df_creditors['Date'], format = "%d/%m/%Y")
# Drop 'Debit' from the DataFrame
df_creditors = df_creditors.drop('Debit', axis=1)

df_debitors = df_transaction[df_transaction['Debit'] != 0].copy()

# Drop 'Debit' from the DataFrame
df_debitors = df_debitors.drop('Credit', axis=1)

unique_details = df_creditors['Details'].unique()
# Create sub DataFrames based on 'Details' values
sub_df_creditors = {}
for detail_value in unique_details:
    sub_df_creditors[detail_value] = df_creditors[df_creditors['Details'] == detail_value]
    
# Display the resulting sub DataFrames

for detail_value, sub_df in sub_df_creditors.items():
    print(f"Sub DataFrame for Details='{detail_value}':")
    print(sub_df)
    print('\n')
    
unique_details = df_debitors['Details'].unique()
# Create sub DataFrames based on 'Details' values
sub_df_debitors = {}
sub_df_cash = pd.DataFrame()
for detail_value in unique_details:
    if "ATM" in detail_value:
        sub_df_cash = pd.concat([sub_df_cash,df_debitors[df_debitors['Details'] == detail_value]], ignore_index = True)
    else:
        sub_df_debitors[detail_value] = df_debitors[df_debitors['Details'] == detail_value]
    
# Display the resulting sub DataFrames
print(sub_df_cash)
for detail_value, sub_df in sub_df_debitors.items():
    print(f"Sub DataFrame for Details='{detail_value}':")
    print(sub_df)
    print('\n')
    
df_creditors['Month'] = df_creditors['Date'].dt.to_period("M")
df_creditors_sum = df_creditors.groupby(['Month','Details']).agg({'Credit': ['sum']})
df_creditors = df_creditors.sort_values(by='Details')
df_debitors = df_debitors.sort_values(by='Details').reset_index()
# print(df_creditors['Date'][0])
#df_creditors = df_creditors.groupby(df_creditors['Details'].agg({'Credit': ['sum']}))
print(df_creditors_sum)
#df_creditors.to_csv('creditor_file.csv', index = False)
#df_debitors.to_csv('debitor_file.csv', index = False)


# In[27]:


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
print(kernel)
final_text_image_without_noise = cv2.erode(final_text_image, kernel, iterations=1)
final_text_image_without_noise = cv2.dilate(final_text_image_without_noise,kernel,iterations=1)
plt.imshow(final_text_image_without_noise)


# In[126]:


kernel_to_remove_gaps_between_words = np.array([
            [1,1,1,1,1,1]
    ])
dilated_words = cv2.dilate(final_text_image_without_noise, kernel_to_remove_gaps_between_words, iterations = 4)
simple_kernel = np.ones((3,3), np.uint8)
dilated_words = cv2.dilate(dilated_words,simple_kernel,iterations = 4)
plt.imshow(dilated_words)


# In[127]:


contours, hierarchy = cv2.findContours(dilated_words,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cordinates.append((x,y,w,h))
    #bounding the images
    if (y< 10000):
        cv2.rectangle(modified_im,(x,y),(x+w,y+h),(0,0,255),1)
plt.imshow(modified_im)
cv2.imwrite('extractedtext.jpg',modified_im)


# In[170]:


contours, hierarchy = cv2.findContours(dilated_vertical_horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cordinates = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cordinates.append((x,y,w,h))
    #bounding the images
    if (y< 10000):
        cv2.rectangle(modified_im,(x,y),(x+w,y+h),(0,0,255),1)
plt.imshow(modified_im)


# In[171]:


cv2.namedWindow('extractedtext.jpg', cv2.WINDOW_NORMAL)
cv2.imwrite('extractedtext.jpg',modified_im)


# In[139]:


from PIL import Image 
image = Image.open("roi.jpg") 
  
right = 100
left = 100
top = 100
bottom = 100
  
width, height = image.size 
  
new_width = width + right + left 
new_height = height + top + bottom 
  
result = Image.new(image.mode, (new_width, new_height), (255, 255, 255)) 
  
result.paste(image, (left, top)) 
  
result.save('output.jpg') 


# In[53]:


plt.imshow(result)


# In[ ]:


import pytesseract
import tabula
import pandas as pd

from pdf2image import convert_from_path
# Path to the Tesseract executable (update this with your path)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
def pdf_to_csv(pdf_path, csv_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    # Perform OCR on each image
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='eng')
        tables = tabula.read_pdf(temp_text_file, pages='all', multiple_tables=True)
    # Concatenate tables into a single DataFrame
    df = pd.concat(tables)
    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
    return text
# Replace 'your_pdf_file.pdf' with the path to your PDF file
pdf_path = "Bank_Statement_Proof_Trial.pdf"
result_text = pdf_to_text(pdf_path)
# Print or save the result
print(result_text)



# In[ ]:


import camelot.io as camelot
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
# Path to the Tesseract executable (update this with your path)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
def pdf_to_csv(pdf_path, csv_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    # Perform OCR on each image
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='eng')
    # Save the extracted text to a temporary text file
    temp_text_file = 'temp_text.txt'
    with open(temp_text_file, 'w', encoding='utf-8') as file:
        file.write(text)
    # Use camelot to extract tables from the PDF
    tables = camelot.read_pdf(temp_text_file, flavor='stream', pages='all')
    # Concatenate tables into a single DataFrame
    df = pd.concat([table.df for table in tables])
    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
# Replace 'your_pdf_file.pdf' and 'output.csv' with the actual paths
pdf_path = 'Bank_Statement_Proof_Trial.pdf'
csv_path = 'output.csv'
pdf_to_csv(pdf_path, csv_path)


# In[ ]:


import tabula
import PyPDF2
import pandas as pd
def pdf_to_csv(pdf_path, csv_path):
    # Use PyPDF2 to get the number of pages in the PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = pdf_reader.numPages
    # Use tabula to extract tables from each page of the PDF
    tables = []
    for page in range(1, num_pages + 1):
        table = tabula.read_pdf(pdf_path, pages=page, multiple_tables=True)
        tables.extend(table)
    # Concatenate tables into a single DataFrame
    df = pd.concat(tables)
    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
# Replace 'your_pdf_file.pdf' and 'output.csv' with the actual paths
pdf_path = 'Bank_Statement_Proof_Trial.pdf'
csv_path = 'output.csv'
pdf_to_csv(pdf_path, csv_path)


# In[ ]:


import tabula
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import os

# Path to the Tesseract executable (update this with your path)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
def pdf_to_csv(pdf_path, csv_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    # Perform OCR on each image
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='eng')
    # Save the extracted text to a temporary text file
    temp_text_file = 'temp_text.txt'
    with open(temp_text_file, 'w', encoding='utf-8') as file:
        file.write(text)
    # Use tabula to extract tables from the PDF
    tables = tabula.read_pdf(temp_text_file, pages='all', multiple_tables=True)
    # Concatenate tables into a single DataFrame
    df = pd.concat(tables)
    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
# Replace 'your_pdf_file.pdf' and 'output.csv' with the actual paths
pdf_path = 'Bank_Statement_Proof_Trial.pdf'
csv_path = 'output.csv'
pdf_to_csv(pdf_path, csv_path)


# In[ ]:


import tabula
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import os

# Path to the Tesseract executable (update this with your path)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
def pdf_to_csv(pdf_path, csv_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    # Perform OCR on each image
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='eng')
    # Save the extracted text to a temporary text file
    temp_text_file = 'temp_text.txt'
    with open(temp_text_file, 'w', encoding='utf-8') as file:
        file.write(text)
    print(text)
    # Use tabula to extract tables from the PDF
    tables = tabula.read_pdf(temp_text_file, pages='all', multiple_tables=True)
    # Concatenate tables into a single DataFrame
    df = pd.concat(tables)
    # Save DataFrame to CSV
    df.to_csv(csv_path, index=False)
# Replace 'your_pdf_file.pdf' and 'output.csv' with the actual paths
pdf_path = 'Bank_Statement_Proof_Trial.pdf'
csv_path = 'output.csv'
pdf_to_csv(pdf_path, csv_path)


# In[ ]:


import tabula
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import os
import csv

# Path to the Tesseract executable (update this with your path)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
def pdf_to_csv(pdf_path, csv_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    # Perform OCR on each image
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='eng')
    # Save the extracted text to a temporary text file
    temp_text_file = 'temp_text.txt'
    with open(temp_text_file, 'w', encoding='utf-8') as file:
        file.write(text)
    print(text)
    # Use tabula to extract tables from the PDF
    text_to_csv(temp_text_file,csv_path)
    # Concatenate tables into a single DataFrame
    
def text_to_csv(text_path, csv_path):
# Read the text file
    with open(text_path, 'r', encoding='utf-8') as text_file:
        text_content = text_file.read()
    # Split the text into lines
    lines = text_content.split('\n')
    # Initialize CSV writer
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Flag to indicate whether we are currently inside a table
        in_table = False
        # Iterate through lines and write to CSV
        for line in lines:
            # Example logic: If a line contains a tab character, consider it a table row
            if '\t' in line:
                # Split the line into cells based on the tab character
                cells = line.split('\t')
                # Write the cells to the CSV file
                csv_writer.writerow(cells)
                # Set in_table flag to True
                in_table = True
            else:
                # If not in a table, check if a table has just ended
                if in_table:
                    in_table = False
                    # Write an empty line to separate tables in the CSV
                    csv_writer.writerow([])

# Replace 'your_pdf_file.pdf' and 'output.csv' with the actual paths
pdf_path = 'Bank_Statement_Proof_Trial.pdf'
csv_path = 'output.csv'
pdf_to_csv(pdf_path, csv_path)


# In[ ]:


def text_to_csv(text_path, csv_path):
    # Read the text file
    with open(text_path, 'r', encoding='utf-8') as text_file:
        text_content = text_file.read()
    # Split the text into lines
    lines = text_content.split('\n')
    # Initialize CSV writer
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Flag to indicate whether we are currently inside a table
        in_table = False
        # Iterate through lines and write to CSV
        for line in lines:
            # Example logic: If a line contains a tab character, consider it a table row
            if '\t' in line:
                # Split the line into cells based on the tab character
                cells = line.split('\t')
                # Write the cells to the CSV file
                csv_writer.writerow(cells)
                # Set in_table flag to True
                in_table = True
            else:
                # If not in a table, check if a table has just ended
                if in_table:
                    in_table = False
                    # Write an empty line to separate tables in the CSV
                    csv_writer.writerow([])
# Replace 'text_file.txt' and 'output.csv' with the actual paths
text_file_path = 'text_file.txt'
csv_output_path = 'output.csv'


# In[ ]:


import pandas as pd

# Specify the column names if needed
# column_names = ['col1', 'col2', 'col3']

# Read the CSV file and handle errors
try:
    df = pd.read_csv("temp_text.txt", sep="|", error_bad_lines=False)
except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")


# In[ ]:


from pdf2image import convert_from_path
import pytesseract
import pandas as pd

# Path to your scanned PDF file
pdf_path = 'Bank_Statement_Proof_Trial.pdf'

# Convert PDF pages to images
pages = convert_from_path(pdf_path)

# OCR and extract text from each image
text_data = []
for i, page in enumerate(pages):
    text = pytesseract.image_to_string(page, lang='eng')
    text_data.append(text)

# Save extracted text to a text file (optional)
with open('extracted_text.txt', 'w', encoding='utf-8') as file:
    file.writelines(text_data)


# In[ ]:


from PIL import Image

# Apply preprocessing to each page
preprocessed_pages = [page.convert('L').point(lambda x: 0 if x < 180 else 255) for page in pages]


# In[ ]:


for i in range(len(preprocessed_pages)):
    preprocessed_pages[i].save(pdf_path + ' -preprocessed' + ' -page' + str(i)+ '.png', 'PNG')


# In[ ]:


import tabula
import pandas as pd

# Replace 'your_pdf_file.pdf' with the actual path to your PDF file
pdf_path = 'Bank_Statement_Proof_Trial.pdf'

# Specify the page number(s) where the table is located
# Use 'pages='all'' to extract tables from all pages
# Example: pages=2 extracts tables from page 2
pages = 'all'

# Extract tables from the specified page(s)
tables = tabula.read_pdf(pdf_path, pages=pages, multiple_tables=True)

# Assuming the last table is the one you want to extract
if len(tables) > 0:
    table_to_extract = tables[-1]
    
    # Display the extracted table
    print(table_to_extract)

    # Save the table to a CSV file
    table_to_extract.to_csv('extracted_table.csv', index=False)
else:
    print("No tables found in the specified page(s).")

