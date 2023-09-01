import tensorflow as tf
import numpy as np
import os
import cv2

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="./converted_tflite_quantized/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
 '  بقعة أوراق الفلفل الحار',
 '  بقعة بكتيرية في الفلفل الحلو',
 ' البقعة المبكرة في البطاطا',
 ' البقعة المبكرة في الطماطم',
 ' التبقع السبتوري لاوراق الطماطم',
 ' التعفن الأسود في العنب',
 ' اللفحة المتأخرة في البطاطا',
 ' المنيا البيضاء للفلفل الحار',
 ' بقعة الهدف في الطماطم',
 ' بقعة بكتيرية في الدراق',
 ' بقعة بكتيرية في الطماطم',
 ' حرق الأوراق في الفراولة',
 ' ذبول الأوراق في العنب ',
 ' صدأ أوراق الذرة الرمادي',
 ' صدأ أوراق الذرة الشمالي',
 ' عفن الأوراق في الطماطم',
 ' عفن بياض الكرز',
 ' فيروس التموج في الطماطم',
 ' فيروس لفحة الأوراق الصفراء في الطماطم',
 ' لفحة أوراق الفلفل الحار',
 '.ipynb_checkpoints',
 'العفن الأسود للتفاح',
 'اللفحة المتأخرة',
 'بثور العنب السوداء',
 'بطاطا سليمة',
 'تفاح سليم',
 'دراق سليم',
 'ذرة سليمة',
 'سوس العنكبوت في الطماطم',
 'صدا التفاح',
 'صدا الذرة',
 'طماطم سليمة',
 'عنب سليم',
 'فراولة سليمة',
 'فلفل حار  مائل للصفرة',
 'فلفل حلو سليم',
 'فلفل سليم',
 'كرز سليم',
 'لفحة التفاح'
]

disease_info = {
    '  بقعة أوراق الفلفل الحار':{
         "نظرة عامة": " هو مرض فطري يسبب تكون المرارة على أوراق وأغصان الزيتون.",
         "الأعراض":"تتكون الكرات الصغيرة المستديرة على الجانب السفلي من أوراق الزيتون. يكون لون الكرات أخضر فاتح في البداية ويتحول إلى اللون البني مع تقدم العمر. يمكن أن تسبب الالتهابات الشديدة سقوط الأوراق.",
         "العلاج": "قم بتقليم وتدمير الفروع المصابة. رش مبيدات الفطريات في الخريف والربيع لمنع المزيد من الانتشار." 
    },
    '  بقعة بكتيرية في الفلفل الحلو':{
        "نظرة عامة":" هو مرض بكتيري يسبب تكون بقع بنية على الفلفل الحلو.",
        "الاعراض":"تظهر الأشجار توقف النمو وأوراقها الذبولية. عند إزالة التربة، تظهر الجذور السوداء والمتحللة.",
        "العلاج":"تقليم وتدمير الأشجار المصابة. ضع جرعات من مبيدات الفطريات في الربيع والخريف لمنع المزيد من الانتشار. قم بتدوير محاصيل التفاح مع النباتات غير المضيفة لتعطيل دورة حياة الفطريات."
        },
    ' البقعة المبكرة في البطاطا': {
        "نظرة عامة": "شجرة تفاح سليمة ولا تظهر عليها علامات المرض أو الآفات.",
        "الاعراض": "الأوراق خضراء ولا تشوبها شائبة. تنضج الثمار بشكل طبيعي دون تشوهات أو تغير في اللون.",
        "العلاج": "عمليات التفتيش والصيانة الدورية حسب الحاجة. لا حاجة للمعالجة بالمبيدات الحشرية أو مبيدات الفطريات."
        },
    ' البقعة المبكرة في الطماطم':{
        "نظرة عامة":"هو مرض فطري يسبب تكون بقع بنية على الطماطم.",
        "الاعراض":"تتشكل بثرات بلون الصدأ على السطح العلوي للورقة في الربيع وأوائل الصيف. الالتهابات الشديدة تسبب سقوط الأوراق.",
        "العلاج":"Prune out and destroy infected branches. Apply fungicide sprays in fall and spring to prevent further spread. Some apple varieties have resistance to specific rust races."
        },
    ' التبقع السبتوري لاوراق الطماطم':{
        "نظرة عامة":"هو مرض فطري يسبب تكون بقع بنية على أوراق الطماطم.",
        "الاعراض":"Olive-green to black spots form on leaves, fruit, and occasionally young twigs. Spots may appear on upper or lower leaf surface. Heavily infected leaves drop prematurely.",
        "العلاج":"Prune out and destroy fallen leaves to reduce inoculum. Apply protective fungicide sprays starting at bud break on a 7-10 day schedule through June. Some apple varieties have resistance to specific scab races."
        },
    ' التعفن الأسود في العنب': {
        "نظرة عامة": "هو مرض فطري يسبب تعفن العنقود والعنب.",
        "الاعراض": "Leaves are green and unblemished. Fruit is ripening normally without deformities or discoloration.",
        "العلاج": "Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
        },
    ' اللفحة المتأخرة في البطاطا':{
        "نظرة عامة":"هو مرض فطري يسبب بقع بنية على البطاطا.",
        "الاعراض":"Brown to black spots develop on upper and lower leaf surfaces. Spots are angular and often have a target-like pattern.",
        "العلاج":"Prune out and destroy infected branches. Apply protective fungicide sprays in spring and fall on a 7-10 day schedule when conditions favor disease."
        },
    ' المنيا البيضاء للفلفل الحار':{
        "نظرة عامة":"هو مرض فطري يسبب تكون بقع بيضاء على الفلفل الحار.",
        "الاعراض":"Water-soaked spots form on leaves which enlarge and become necrotic. Similar spots form on fruit. Heavy infections cause defoliation.",
        "العلاج":"Remove volunteer plants and crop debris to reduce inoculum. Apply copper bactericide sprays on a 7-10 day schedule when conditions favor disease. Use resistant varieties when available."
        },
    ' بقعة الهدف في الطماطم': {
        "نظرة عامة": "فلفل حلو صحي لا تظهر عليه علامات المرض أو الآفات.",
        "الاعراض": "Leaves are green and unblemished. Fruit is ripening normally without deformities or discoloration.",
        "العلاج": "Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
        },
    ' بقعة بكتيرية في الدراق':{
        "نظرة عامة":"هو مرض بكتيري يسبب تكون بقع بنية على أوراق الدراق.",
        "الاعراض":"Brown spots with concentric rings form on leaves. Similar lesions form on stems. Heavily infected leaves turn yellow and die.",
        "العلاج":"Remove volunteer plants and crop debris to reduce inoculum. Apply protective fungicide sprays starting at first appearance of symptoms on a 7-10 day schedule."
        },
    ' بقعة بكتيرية في الطماطم': {
        "نظرة عامة": "هو مرض بكتيري يسبب تكون بقع بنية على الطماطم.",
        "الاعراض": "Leaves are green and unblemished. Tubers are developing normally underground without rot or deformities.",
        "العلاج": "Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
        },
    ' حرق الأوراق في الفراولة':{
        "نظرة عامة":"هو مرض فطري يسبب حرق وتقشير أوراق الفراولة.",
        "الاعراض":"Dark green to black, water-soaked lesions form on leaves. Similar lesions form on stems and fruits. Lesions enlarge rapidly under humid conditions.",
        "العلاج":"Remove volunteer plants and crop debris to reduce inoculum. Apply protective fungicides starting at first appearance of symptoms and continue on a 5-7 day schedule."
        },
    ' ذبول الأوراق في العنب ':{
        "نظرة عامة":"هو مرض يسبب ذبول وسقوط أوراق العنب.",
        "الاعراض":"Small, circular spots form on leaves with concentric rings. Similar spots develop on fruit. Heavily infected leaves turn yellow and drop.",
        "العلاج":"Remove volunteer plants and crop debris to reduce inoculum. Apply protective fungicides starting at first appearance of symptoms and continue on a 7-10 day schedule."
        },
    ' صدأ أوراق الذرة الرمادي': {
        "نظرة عامة": "هو مرض فطري يسبب بقع رمادية على أوراق الذرة.",
        "الاعراض": "Leaves show light and dark green mosaic or mottling patterns. Fruits develop greenish deformities and are smaller than normal. Disease spreads through contact with infected plants or tools.",
        "العلاج": "Remove and destroy infected plants. Do not reuse pots or tools that contacted infected plants without disinfecting. Some varieties have resistance to specific strains."
        },
    ' صدأ أوراق الذرة الشمالي': {
        "نظرة عامة": "هو مرض فطري يسبب بقع بنية على أوراق الذرة.",
        "الاعراض": "Young leaves become yellow, thickened and curled upwards. Flowers and fruits often fail to develop. Disease spreads by whitefly feeding.",
        "العلاج": "Control whiteflies with insecticide sprays. Remove and destroy infected plants. Some varieties have resistance to specific strains."
        },
    ' عفن الأوراق في الطماطم':{
        "نظرة عامة":"هو مرض فطري يسبب عفونة وسقوط أوراق الطماطم.",
        "الاعراض":"Water-soaked spots form on leaves which enlarge and become necrotic. Similar spots form on fruit. Heavy infections cause defoliation.",
        "العلاج":"Remove volunteer plants and crop debris to reduce inoculum. Apply copper bactericide sprays on a 7-10 day schedule when conditions favor disease. Use resistant varieties when available."
        },
    ' عفن بياض الكرز':{
        "نظرة عامة":"هو مرض فطري يسبب تكون بقع بيضاء على ثمار الكرز.",
        "الاعراض":"Brown spots with concentric rings form on leaves. Similar lesions form on stems. Heavily infected leaves turn yellow and die.",
        "العلاج":"Remove volunteer plants and crop debris to reduce inoculum. Apply protective fungicide sprays starting at first appearance of symptoms on a 7-10 day schedule."
        },
    ' فيروس التموج في الطماطم': {
        "نظرة عامة": "هو فيروس يسبب تشوهات وتموجات على أوراق وثمار الطماطم.",
        "الاعراض": "Leaves are green and unblemished. Fruits are ripening normally without rot or deformities.",
        "العلاج": "Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
        },
    ' فيروس لفحة الأوراق الصفراء في الطماطم':{
        "نظرة عامة":"هو فيروس يسبب لفحة وتصبغ الأوراق باللون الأصفر.",
        "الاعراض":"Grayish-green spots form on leaves. Spots enlarge and coalesce causing blighting of leaves. Gray fuzzy growth may develop on lower leaf surfaces in humid conditions.",
        "العلاج":"Remove volunteer plants and crop debris to reduce inoculum. Apply protective fungicides starting at first appearance of symptoms and continue on a 7-10 day schedule."
        },
    ' لفحة أوراق الفلفل الحار': {
        "نظرة عامة": "هو مرض فطري يسبب لفحة وتساقط أوراق الفلفل الحار.",
        "الاعراض": "Small, circular to irregular dark spots form on leaves. Spots may enlarge and coalesce during wet weather. Heavily infected leaves turn yellow and drop.",
        "العلاج": "Remove volunteer plants and crop debris to reduce inoculum. Apply protective fungicides starting at first appearance of symptoms and continue on a 7-10 day schedule."
        },
    '.ipynb_checkpoints':{
        "نظرة عامة": "هو مرض فطري يسبب تكون بقع سوداء على ثمار التفاح.",
        "الاعراض": "Dark sunken lesions form on fruit especially near stem end. Lesions enlarge and become covered with black fungal growth. Twigs and leaves may also be infected.",
        "العلاج": "Remove and destroy infected fruit and fallen leaves. Apply protective fungicide sprays starting at bud break and continuing throughout the growing season on a 7-10 day schedule."
    },
    'العفن الأسود للتفاح': {
        "نظرة عامة": "هو مرض فطري يسبب تكون بقع سوداء على ثمار التفاح.",
        "الاعراض": "Dark sunken lesions form on fruit especially near stem end. Lesions enlarge and become covered with black fungal growth. Twigs and leaves may also be infected.",
        "العلاج": "Remove and destroy infected fruit and fallen leaves. Apply protective fungicide sprays starting at bud break and continuing throughout the growing season on a 7-10 day schedule."
    },
    'اللفحة المتأخرة':{
        "نظرة عامة":"Late blight is a fungal disease that causes leaf blight and rotting of tomato fruits and potato tubers. It spreads rapidly under cool, wet conditions.",
        "الاعراض":"Dark green to black, water-soaked lesions form on leaves. Similar lesions form on stems and fruits. Entire plants may be blighted during severe outbreaks.",
        "العلاج":"Remove volunteer plants and crop debris to reduce inoculum. Apply protective fungicides starting at first appearance of symptoms and continue on a 7-10 day schedule. Use resistant varieties when available."
    },
    'بثور العنب السوداء':{
        "نظرة عامة":"Black rot of grapes is a fungal disease that causes dark, sunken lesions on fruit clusters and shoots. It overwinters in infected canes and spreads during wet weather.",
        "الاعراض":"Dark brown to black, sunken lesions form on fruit clusters. Lesions enlarge and berries rot. Similar lesions form on shoots and young canes.",
        "العلاج":"Prune out and destroy infected canes in winter. Apply protective fungicide sprays starting at bud break and continuing throughout the growing season on a 7-10 day schedule."
    },
    'بطاطا سليمة':{
        "نظرة عامة":"A healthy potato plant showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Tubers are forming normally underground without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'تفاح سليم':{
        "نظرة عامة":"A healthy apple tree showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Fruits are ripening normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'دراق سليم':{
        "نظرة عامة":"A healthy grapevine showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Fruit clusters are forming normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'ذرة سليمة':{
        "نظرة عامة":"A healthy corn plant showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Ears are forming normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'سوس العنكبوت في الطماطم':{
        "نظرة عامة":"Two spotted spider mites cause stippling and yellowing of tomato leaves. Heavy infestations cause leaf drop and reduced yield.",
        "الاعراض":"Leaves show small yellow or whitish stippling. Webbing may be seen on undersides of leaves with mites and eggs present. Heavily infested leaves turn yellow, brown and drop prematurely.",
        "العلاج":"Control with miticide sprays when mites first appear. Repeat applications may be needed. Release predatory mites where possible to maintain control."
    },
    'صدا التفاح':{
        "نظرة عامة":"Sooty blotch and flyspeck are fungal diseases that cause black spots on apple fruit. They do not rot fruit but reduce quality and marketability.",
        "الاعراض":"Sooty blotch appears as irregular, sooty black patches on fruit surface. Flyspeck appears as tiny black dots scattered over fruit surface.",
        "العلاج":"Apply protective fungicide sprays starting at tight cluster and continuing at 7-10 day intervals through harvest. Good coverage and thorough drying of fruit is important for control."
    },
    'صدا الذرة':{
        "نظرة عامة":"Smut of corn is a fungal disease that causes galls to form on leaves, stalks, ears and tassels. It overwinters in corn debris and soil.",
        "الاعراض":"Grey or black powdery spore masses form in swollen galls on any above ground part of the plant. Galls rupture to release spores that spread the disease.",
        "العلاج":"Remove and destroy corn stalk residues after harvest. Rotate crops and use certified seed treated with fungicide. Foliar fungicides may help reduce spread if applied preventatively."
    },
    'طماطم سليمة':{
        "نظرة عامة":"A healthy tomato plant showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Fruits are ripening normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'عنب سليم':{
        "نظرة عامة":"A healthy grapevine showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Fruit clusters are forming normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'فراولة سليمة':{
        "نظرة عامة":"A healthy strawberry plant showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Fruits are ripening normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'فلفل حار  مائل للصفرة':{
        "نظرة عامة":"A pepper plant showing early signs of nutrient deficiency",
        "الاعراض":"New growth is stunted. Older leaves are yellowing between the veins while younger leaves are pale green or yellow.",
        "العلاج":"Apply a balanced fertilizer according to soil test results. Increase organic matter in soil with compost or manure. Maintain adequate but not excessive moisture."
    },
    'فلفل حلو سليم':{
        "نظرة عامة":"A healthy sweet pepper plant showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Fruits are ripening normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'فلفل سليم':{
        "نظرة عامة":"A healthy bell pepper plant showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Fruits are ripening normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'كرز سليم':{
        "نظرة عامة":"A healthy cherry tree showing no signs of disease or pests.",
        "الاعراض":"Leaves are green and unblemished. Fruit clusters are forming normally without rot or deformities.",
        "العلاج":"Regular inspections and maintenance as needed. No pesticide or fungicide treatment required."
    },
    'لفحة التفاح':{
        "نظرة عامة":"Apple scab is a fungal disease that causes olive green to black lesions on leaves and fruit. It overwinters in fallen leaves and spreads during wet weather in spring.",
        "الاعراض":"Olive green to black lesions form on leaves. Lesions may coalesce and cause leaf drop. Lesions also form on fruit making them unmarketable.",
        "العلاج":"Apply protective fungicide sprays starting at green tip and continuing at 7-10 day intervals through primary scab season. Good coverage and thorough drying of fruit is important for control. Remove and destroy fallen leaves to reduce inoculum."
    },


}



def classify_image_with_info(image_path):
    # Load and preprocess the image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img = img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get the classification results
    output = interpreter.get_tensor(output_details[0]['index'])

    class_index = np.argmax(output)
    class_name = class_names[class_index]
    
    # Get disease details and treatment recommendations
    disease_details = disease_info.get(class_name, {})
    
    return class_name, disease_details

# # Example image path
# image_path = "./images_test/healthy_olive.jpg"

# # Perform classification and get disease details
# class_name, disease_details = classify_image_with_info(image_path)

# print("Predicted class:", class_name)
# print("Disease Details:", disease_details)
