from pdf2image import convert_from_path
import pandas as pd
import dateutil.parser
import numpy as np
import re
import cv2
import pytesseract


def detect_table(img):
    #im1 = cv2.imread(file, 0)
    #im = cv2.imread(file)
    im = img
    im1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    return intermediate_img

def convert_image_to_table(intermediate_img):
    modified_im = intermediate_img
    modified_im1 = cv2.cvtColor(intermediate_img,cv2.COLOR_BGR2GRAY)
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
            results_from_ocr = pytesseract.image_to_string(cropped_image, lang='eng').strip()
            results_from_ocr = results_from_ocr.replace("\n", " ")
            results_from_ocr = results_from_ocr.replace(",", "")
            current_row.append(results_from_ocr)
            image_number += 1
        table.append(current_row)
        current_row = []
    header = table[0]
    return table, header

def convert_str_to_date(date_list):
    final_output=[]
    for i in date_list:
        try:
            final_output.append(dateutil.parser.parse(i.replace("!","")).strftime("%d/%m/%Y"))
        except:
            final_output.append(np.nan)
    return final_output

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


final_table = []
initial_header = []
current_header = []

# Provide the path to the PDF file you want to convert
pdf_path = "10000923_BANK_STMT_1.pdf"

# Enclose the file path in double quotes to handle spaces
images = convert_from_path(pdf_path)

for i in range(len(images)):
    img = np.array(images[i])
    img = img[:, :, ::-1].copy()
    intermediate_img = detect_table(img)
    converted_table, header = convert_image_to_table(intermediate_img)
    if i==0:
        initial_header = header
    else:
        current_header = header
    if current_header == initial_header:
        final_table.append(converted_table[1:])
    else:
        final_table.append(converted_table)

with open("final_output.csv", "w") as f:
    for converted_table in final_table:
        for row in converted_table:
            f.write(",".join(row) + "\n")

df = pd.read_csv("final_output.csv")
df = df.rename(columns = {'Txn Date':'Date'})
txn_date = convert_str_to_date(df['Date'].tolist())
df['Date'] = txn_date
df = df.dropna(subset = ['Date'])
df['Balance'] = df['Balance'].apply(clean_and_convert)
df['Credit'] = df['Credit'].apply(clean_and_convert)
df['Debit'] = df['Debit'].apply(clean_and_convert)


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

for detail_value, sub_df in sub_df_debitors.items():
    print(f"Sub DataFrame for Details='{detail_value}':")
    print(sub_df)
    print('\n')

df_creditors['Month'] = df_creditors['Date'].dt.to_period("M")
df_creditors_sum = df_creditors.groupby(['Month','Details']).agg({'Credit': ['sum']})
df_creditors = df_creditors.sort_values(by='Details')
df_debitors = df_debitors.sort_values(by='Details').reset_index()