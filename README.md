# MSME Sector Mapper for Targeted Policy Implementation and Market Analysis

## Short Description

This novel approach addresses the need for accurate classification and analysis of Micro, Small, and Medium Enterprises (MSMEs) based on their business activities. The problem is important because governments and policymakers rely on precise sector classification to design targeted interventions, allocate resources effectively, and understand economic trends. By automatically mapping enterprises to standardized NIC (National Industrial Classification) codes, this system enables more efficient policy implementation, market analysis, and economic planning.

Our approach achieves impressive results with up to **95.3% accuracy** in classifying MSMEs into their appropriate industrial sectors, providing a robust foundation for data-driven decision making in economic development.

---

## Dataset Source

- **Dataset Size**: 50,000 MSME records from Maharashtra, India  
- **Original Features**: 9 columns including enterprise names, addresses, registration dates, and activity descriptions  
- **Key Challenge**: The activity information was stored in JSON-like string format requiring specialized extraction  

---

## Preprocessing Steps
- Extracted activity descriptions and NIC codes from nested JSON structures  
- Handled missing values (33 missing NIC codes, 1 missing enterprise name)  
- Filtered out rare classes with fewer than 10 samples to improve model stability  
- Merged enterprise names and activity descriptions into a unified text feature  
- Applied comprehensive text cleaning including lowercasing, punctuation removal, and lemmatization  

---

## Approach 
I employed a multi-class text classification approach using TF-IDF vectorization combined with traditional machine learning classifiers. This approach was chosen because:

1. **Interpretability**: Traditional ML models provide transparent decision-making processes crucial for policy applications  
2. **Computational Efficiency**: Faster training and inference compared to deep learning alternatives  
3. **Data Efficiency**: Effective with moderate dataset sizes without requiring extensive computational resources  
 

## Steps to Run the Code

1. **Environment Setup**:
    ```bash
    pip install pandas numpy scikit-learn nltk transformers
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
2. **Data Preparation**:

   - Place the msme_MAHARASHTRA.csv file in the specified directory path
   - Update the file path in the notebook:

    ```bash
    pd.read_csv(f"E:\\ML proj\\Data\\msme_MAHARASHTRA.csv")
3. **Execution**:

   - Run the Jupyter notebook cells sequentially from top to bottom
   - The code automatically handles all preprocessing, training, and evaluation

4. **Key Dependencies**:

    - pandas, numpy, scikit-learn
    - nltk for text preprocessing
    - transformers (Hugging Face) for tokenizer 

---

## Results Summary

### Model Performance Comparison

| Model | Accuracy | F1-score | MSE | MAE | RMSE |
|-------|----------|----------|-----|-----|------|
| Logistic Regression | 91.76% | 91.19% | 5954.19 | 14.13 | 77.16 |
| Random Forest | 93.40% | 93.03% | 5365.45 | 12.49 | 73.25 |
| **SVM** | **95.33%** | **95.29%** | **4715.19** | **10.38** | **68.67** |

### Performance Metrics Interpretation

- **Accuracy (91.76%-95.33%)**: Out of 100 enterprises, approximately 92-95 are correctly classified to their right NIC code sector
- **F1-score (91.19%-95.29%)**: The models maintain excellent balance between finding all relevant enterprises (recall) and only selecting correct ones (precision)
- **MSE (4715.19-5954.19)**: The average squared prediction error ranges from 4715 to 5954 squared class units
- **MAE (10.38-14.13)**: On average, predicted class labels are about 10-14 class units away from the true labels
- **RMSE (68.67-77.16)**: The typical prediction error ranges from 69-77 class units in the original scale
---
## 📊 Results & Visualizations

### MSME Data Analysis Results

| Word Cloud Analysis | Activity Distribution |
|:-------------------:|:---------------------:|
| <img src="images/wordcloud.png" width="400"> | <img src="images/activity_distribution.png" width="400"> |
| Shows most frequent activities and enterprise names | Displays distribution across business sectors |

Word Cloud Analysis
![Word Cloud](images/wordcloud.png)
The word cloud visualization highlights the most common business activities and enterprise names...

---
### Key Findings
1. **SVM Superiority**: The LinearSVC model achieved the best performance across all metrics:
   - Highest classification accuracy (95.33%)
   - Lowest error rates (MSE: 4715.19, MAE: 10.38, RMSE: 68.67)
   - Best F1-score (95.29%) indicating balanced precision and recall

2. **Error Analysis**: The RMSE values show that on average, the model predictions are within 68-77 class units of the true labels, which is reasonable given the multi-class nature of the problem.

3. **Progressive Improvement**: Each model shows consistent improvement over the previous one:
   - Random Forest reduced MSE by ~10% compared to Logistic Regression
   - SVM further reduced MSE by ~12% compared to Random Forest

## Conclusion

This project demonstrates that traditional ML with careful feature engineering achieves **95.3% accuracy** in MSME sector classification. Key outcomes:

- **High Accuracy**: Reliable NIC code mapping for precise sector analysis
- **Policy Ready**: Actionable intelligence for targeted economic planning  
- **Scalable**: Easily extendable to larger datasets and regions
- **Efficient**: Suitable for real-time government portal implementation

**Future Work**: Explore hybrid approaches combining traditional ML interpretability with transformer models for challenging cases.

## References

[1] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[2] S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. O'Reilly Media, 2009.

[3] Ministry of Statistics and Programme Implementation, Government of India, "National Industrial Classification (NIC)," 2008.

[4] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[5] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
