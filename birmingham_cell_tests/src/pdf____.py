#!/usr/bin/env python3
import fitz  # PyMuPDF

# Funzione per aggiungere una nuova pagina con testo a un PDF esistente
def add_page_with_text_to_pdf(existing_pdf_path, text, output_pdf_path, position=(72, 72), font_size=12):
    # Apri il PDF esistente
    doc = fitz.open(existing_pdf_path)
    
    # Crea una nuova pagina alla fine del documento
    # new_page = doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)
    new_page = doc.load_page(0)
    # Imposta la posizione del testo e il font
    text_rect = fitz.Rect(position[0], position[1], new_page.rect.width - 72, new_page.rect.height - 72)
    new_page.insert_textbox(text_rect, text, fontsize=font_size)
    
    # Salva il PDF modificato
    doc.save(output_pdf_path)

# Percorso del PDF esistente e del PDF di output
input_pdf_path = "/home/gauss/Downloads/Michelle_Delledonne_Attendance_Cerificate_June_2024.pdf"
existing_pdf_path = "/home/gauss/Downloads/aaa.pdf"
output_pdf_path = "/home/gauss/Downloads/add.pdf"

doc = fitz.open(input_pdf_path)
page = doc.load_page(0)
text_to_add = page.get_text()

# Testo da aggiungere
# text_to_add = "Questo è il testo aggiunto sulla nuova pagina."
print(text_to_add)
print(type(text_to_add))
# Aggiungi la nuova pagina con il testo al PDF esistente
add_page_with_text_to_pdf(existing_pdf_path, text_to_add[:25], output_pdf_path)

print("Nuova pagina con testo aggiunta al PDF esistente con successo!")

exit()
from PyPDF2 import PdfReader
import fitz  # PyMuPDF

# Funzione per estrarre il testo da un PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() + "\n"
    return text

# Funzione per aggiungere testo a un PDF esistente
def add_text_to_existing_pdf(existing_pdf_path, text, output_pdf_path):
    doc = fitz.open(existing_pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(text_rect, text, fontsize=12)
    doc.save(output_pdf_path)

# Percorso del PDF di input e del PDF esistente
input_pdf_path = "/home/gauss/Downloads/Michelle_Delledonne_Attendance_Cerificate_June_2024.pdf"
existing_pdf_path = "/home/gauss/Downloads/aaa.pdf"
output_pdf_path = "/home/gauss/Downloads/add.pdf"

# Estrai il testo dal PDF di input    
reader = PdfReader(input_pdf_path)
text = ""

a_page = reader.pages[0]
text = a_page.extract_text()

doc = fitz.open(input_pdf_path)
page = doc.load_page(0)
text += page.get_text() + "\n"
print(text)

# Aggiungi il testo estratto al PDF esistente
doc = fitz.open(existing_pdf_path)
page = doc.load_page(0)
text_rect = fitz.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
page.insert_textbox(text_rect, text, fontsize=12)
doc.save(output_pdf_path)
text_rect = fitz.Rect(position[0], position[1], page.rect.width - 72, page.rect.height - 72)

print("Testo aggiunto al PDF esistente con successo!")

exit()

from PyPDF2 import PdfReader, PdfWriter

input_pdf_path = "/home/gauss/Downloads/Michelle_Delledonne_Attendance_Cerificate_June_2024.pdf"
output_pdf_path = "/home/gauss/Downloads/Michelle_Delledonne_Attendance_Cerificate_June_2024_2.pdf"

# Carica il PDF esistente
reader = PdfReader(input_pdf_path)
writer = PdfWriter()

# Sostituisci testo in una pagina specifica (limitato)
for page_num in range(len(reader.pages)):
    page = reader.pages[page_num]
    text = page.extract_text()
    print(text)
    if "student at the University of Birmingham from June  1st to June 30th," in text:
        print('trovato!!!')
        text = text.replace("student at the University of Birmingham from June  1st to June 30th,", 
                            "student at the University of Birmingham from January 8th to June 30th,")
        # PyPDF2 non permette di modificare il testo direttamente, quindi bisogna ricreare il contenuto
    page.add_text

# Aggiungi le pagine modificate al writer
for page in reader.pages:
    writer.add_page(page)

# Salva il PDF modificato
with open(output_pdf_path, "wb") as output_pdf:
    writer.write(output_pdf)

