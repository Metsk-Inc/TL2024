import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, silhouette_score, confusion_matrix
import plotly.figure_factory as ff

def check_data_requirements(data):
    if len(data.columns) < 2:
        return False, "Ο πίνακας δεδομένων πρέπει να περιέχει τουλάχιστον δύο στήλες (χαρακτηριστικά και ετικέτες)."

    expected_output_column = data.columns[-1]
    if "label" not in expected_output_column.lower():
        return False, "Η τελευταία στήλη του πίνακα δεδομένων πρέπει να περιέχει τις ετικέτες (labels)."

    return True, "Ο πίνακας δεδομένων πληροί τις προδιαγραφές."

def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    z = cm
    x = ["Predicted " + str(i) for i in range(cm.shape[1])]
    y = ["Actual " + str(i) for i in range(cm.shape[0])]
    z_text = [[str(y) for y in x] for x in z]

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)

def classification_tab(X, y):
    st.subheader("Αλγόριθμοι Κατηγοριοποίησης")

    # Διαχωρισμός δεδομένων σε εκπαίδευση και δοκιμή
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    st.write("Ακρίβεια Logistic Regression:", lr_accuracy)
    plot_confusion_matrix(y_test, lr_pred, "Confusion Matrix για Logistic Regression")

    # Support Vector Machine (SVM)
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    st.write("Ακρίβεια SVM:", svm_accuracy)
    plot_confusion_matrix(y_test, svm_pred, "Confusion Matrix για SVM")

    return lr_accuracy, svm_accuracy

def plot_cluster_scatter(X, labels, title):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    fig = px.scatter(x=principal_components[:, 0], y=principal_components[:, 1], color=labels.astype(str), title=title)
    st.plotly_chart(fig, use_container_width=True)

def clustering_tab(X):
    st.subheader("Αλγόριθμοι Ομαδοποίησης")

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    kmeans_labels = kmeans.labels_
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    st.write("Σκορ Silhouette για KMeans:", kmeans_silhouette)
    plot_cluster_scatter(X, kmeans_labels, "KMeans Clustering")

    # DBSCAN
    dbscan = DBSCAN()
    dbscan.fit(X)
    dbscan_labels = dbscan.labels_
    if len(set(dbscan_labels)) > 1:  # Έλεγχος αν υπάρχουν περισσότερες από μία κατηγορίες
        dbscan_silhouette = silhouette_score(X, dbscan_labels)
    else:
        dbscan_silhouette = -1  # Αποφυγή λάθους όταν όλες οι ετικέτες είναι ίδιες
    st.write("Σκορ Silhouette για DBSCAN:", dbscan_silhouette)
    plot_cluster_scatter(X, dbscan_labels, "DBSCAN Clustering")

    return kmeans_silhouette, dbscan_silhouette

def results_comparison_tab(classification_results, clustering_results):
    st.subheader("Αποτελέσματα και Σύγκριση Αλγορίθμων")
    lr_accuracy, svm_accuracy = classification_results
    kmeans_silhouette, dbscan_silhouette = clustering_results

    st.write("**Ακρίβεια Κατηγοριοποίησης:**")
    st.write("- Logistic Regression:", lr_accuracy)
    st.write("- SVM:", svm_accuracy)

    st.write("**Σκορ Silhouette Ομαδοποίησης:**")
    st.write("- KMeans:", kmeans_silhouette)
    st.write("- DBSCAN:", dbscan_silhouette)

def eda_tab(data):
    st.subheader("Exploratory Data Analysis (EDA)")

    st.write("Περιγραφή δεδομένων:")
    st.write(data.describe())

    st.write("Κατανομή ετικετών:")
    st.write(data.iloc[:, -1].value_counts())

    fig1 = px.histogram(data, x=data.columns[0])
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(data, x=data.columns[1])
    st.plotly_chart(fig2, use_container_width=True)

def info_tab():
    st.subheader("Πληροφορίες")
    st.write("Αυτή η εφαρμογή δημιουργήθηκε για την ανάλυση δεδομένων και την εκπαίδευση μοντέλων μηχανικής μάθησης απο τους Ιορδάνη Τσουτσούλη Π2019117,Κωνσταντίνο Ζωγράφο inf2021058,Βασίλειο Λάμπα inf2021118.")
    st.write("**Ομάδα Ανάπτυξης**")
    st.write("- Ο κώδικας αυτός αποτελεί μια ολοκληρωμένη εφαρμογή ανάλυσης δεδομένων και μηχανικής μάθησης, αναπτυγμένη σε Python χρησιμοποιώντας την πλατφόρμα Streamlit. Η εφαρμογή επιτρέπει τη φόρτωση δεδομένων από αρχεία CSV ή Excel και διενεργεί έναν αρχικό έλεγχο για να διασφαλίσει ότι τα δεδομένα πληρούν τις απαιτούμενες προδιαγραφές. Ειδικότερα, τα δεδομένα πρέπει να περιέχουν τουλάχιστον δύο στήλες, με την τελευταία στήλη να περιλαμβάνει τις ετικέτες (labels). Η εφαρμογή προσφέρει τρεις κύριες λειτουργίες μέσω διακριτών καρτελών: 2D οπτικοποίηση δεδομένων χρησιμοποιώντας αλγορίθμους μείωσης διάστασης όπως PCA και t-SNE, εκτέλεση αλγορίθμων κατηγοριοποίησης (Logistic Regression και SVM) και ομαδοποίησης (KMeans και DBSCAN), και παρουσίαση των αποτελεσμάτων των αλγορίθμων μαζί με συγκριτικές μετρήσεις απόδοσης. Επιπλέον, η εφαρμογή περιλαμβάνει μια καρτέλα για εξερεύνηση δεδομένων (EDA) και μια καρτέλα πληροφοριών που παρέχει λεπτομέρειες σχετικά με την εφαρμογή και την ομάδα ανάπτυξης. Για τις δοκιμές, δημιουργήθηκαν τρία αρχεία CSV με δεδομένα κατηγοριοποίησης και ομαδοποίησης, τα οποία πληρούν τις προδιαγραφές και εξασφαλίζουν την ομαλή λειτουργία της εφαρμογής.")
    st.write("- Για την εργασία αυτή λειτουργήσαμε ομαδικά,ωστόσο προσπαθήσαμε να αναθέσουμε σε κάθε έναν από εμας ενα κομμάτι της, και στην συνέχεια όλοι μαζί  εξετάζαμε την λύση που βρήκε.Ειδκότερα  ο Ιορδάνης Τσουτσούλης ασχολήθηκε με το διάβασμα των csv και excel αρχείων.Ο κωνσταντινος Ζωγράφος ανέλαβε να βρει ποιούς αλγορίθμους clustering και ομαδοποίησης θα χρησιμοποιήσουμε ενώ ο Βασίλειος Λάμπας ασχολήθηκε με τις τεχνικές οπτικοποίησης.Τέλος αποφασίσαμε να χρησιμοποιήσουμε το streamlit διότι έχουμε μια επαφή μαζί του απο το μάθημα αναγνώρισης προτύπων. ")

def main():
    st.title("Εφαρμογή Ανάλυσης Δεδομένων")

    uploaded_file = st.file_uploader("Φορτώστε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Εμφάνιση δεδομένων:")
        st.write(data)

        passed, message = check_data_requirements(data)
        if passed:
            st.success(message)
        else:
            st.error(message)
            return  # Σταματήστε την εκτέλεση αν τα δεδομένα δεν πληρούν τις προδιαγραφές

        # Διαχωρισμός χαρακτηριστικών και ετικετών
        X = data.iloc[:, :-1]  # Όλες οι στήλες εκτός από την τελευταία
        y = data.iloc[:, -1]  # Η τελευταία στήλη

        # Χρησιμοποιούμε tabs για να διαχωρίσουμε τις ενότητες
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["2D Visualization", "Κατηγοριοποίηση", "Ομαδοποίηση", "Αποτελέσματα και Σύγκριση", "Πληροφορίες"])

        with tab1:
            st.subheader("2D Visualization")
            # Ανάλυση PCA στα χαρακτηριστικά
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X)

            fig = px.scatter(x=principal_components[:, 0], y=principal_components[:, 1], color=y.astype(str), title="PCA Analysis")
            st.plotly_chart(fig, use_container_width=True)

            # Ανάλυση TSNE στα χαρακτηριστικά με perplexity μικρότερο από τον αριθμό των δειγμάτων
            tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1))  # Χρησιμοποιούμε το min(30, len(X) - 1)
            tsne_components = tsne.fit_transform(X)

            fig = px.scatter(x=tsne_components[:, 0], y=tsne_components[:, 1], color=y.astype(str), title="TSNE Analysis")
            st.plotly_chart(fig, use_container_width=True)

            eda_tab(data)

        with tab2:
            classification_results = classification_tab(X, y)

        with tab3:
            clustering_results = clustering_tab(X)

        with tab4:
            results_comparison_tab(classification_results, clustering_results)

        with tab5:
            info_tab()

if __name__ == "__main__":
    main()
