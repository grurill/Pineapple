import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QFileDialog, QComboBox, QListWidget, QLineEdit, QTabWidget, QTextEdit, 
                             QMessageBox, QCheckBox, QSpinBox, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineeringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Feature Engineering App')
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QHBoxLayout()

        # Left panel for data loading and column selection
        left_panel = QVBoxLayout()
        self.load_button = QPushButton('Load CSV')
        self.load_button.clicked.connect(self.load_csv)
        left_panel.addWidget(self.load_button)

        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.ExtendedSelection)
        left_panel.addWidget(QLabel('Columns:'))
        left_panel.addWidget(self.column_list)

        # Right panel with tabs for different operations
        right_panel = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_missing_values_tab(), "Missing Values")
        self.tabs.addTab(self.create_encoding_tab(), "Encoding")
        self.tabs.addTab(self.create_scaling_tab(), "Scaling")
        self.tabs.addTab(self.create_binning_tab(), "Binning")
        self.tabs.addTab(self.create_transformation_tab(), "Transformation")
        self.tabs.addTab(self.create_feature_selection_tab(), "Feature Selection")
        self.tabs.addTab(self.create_visualization_tab(), "Visualization")
        self.tabs.addTab(self.create_outlier_detection_tab(), "Outliers")
        right_panel.addWidget(self.tabs)

        # Apply and Save buttons
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton('Apply Changes')
        self.apply_button.clicked.connect(self.apply_changes)
        self.save_button = QPushButton('Save CSV')
        self.save_button.clicked.connect(self.save_csv)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.save_button)
        right_panel.addLayout(button_layout)

        # Preview area
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        right_panel.addWidget(QLabel('Data Preview:'))
        right_panel.addWidget(self.preview)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 2)
        self.setLayout(main_layout)

    def create_missing_values_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.missing_strategy = QComboBox()
        self.missing_strategy.addItems(['Drop', 'Fill with Mean', 'Fill with Median', 'Fill with Mode', 'Fill with Value', 'Fill with Most Frequent'])
        layout.addWidget(QLabel('Strategy:'))
        layout.addWidget(self.missing_strategy)

        self.fill_value = QLineEdit()
        self.fill_value.setPlaceholderText('Fill value (if applicable)')
        layout.addWidget(self.fill_value)

        tab.setLayout(layout)
        return tab

    def create_encoding_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.encoding_strategy = QComboBox()
        self.encoding_strategy.addItems(['Label Encoding', 'One-Hot Encoding'])
        layout.addWidget(QLabel('Encoding Method:'))
        layout.addWidget(self.encoding_strategy)

        tab.setLayout(layout)
        return tab

    def create_scaling_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.scaling_strategy = QComboBox()
        self.scaling_strategy.addItems(['StandardScaler', 'MinMaxScaler'])
        layout.addWidget(QLabel('Scaling Method:'))
        layout.addWidget(self.scaling_strategy)

        tab.setLayout(layout)
        return tab

    def create_binning_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.bin_strategy = QComboBox()
        self.bin_strategy.addItems(['Equal Width', 'Equal Frequency'])
        layout.addWidget(QLabel('Binning Strategy:'))
        layout.addWidget(self.bin_strategy)

        self.num_bins = QSpinBox()
        self.num_bins.setRange(2, 100)
        self.num_bins.setValue(5)
        layout.addWidget(QLabel('Number of Bins:'))
        layout.addWidget(self.num_bins)

        tab.setLayout(layout)
        return tab

    def create_transformation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.transform_strategy = QComboBox()
        self.transform_strategy.addItems(['Log', 'Square Root', 'Box-Cox'])
        layout.addWidget(QLabel('Transformation:'))
        layout.addWidget(self.transform_strategy)

        tab.setLayout(layout)
        return tab

    def create_feature_selection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.target_column = QLineEdit()
        self.target_column.setPlaceholderText('Target Column')
        layout.addWidget(QLabel('Target Column:'))
        layout.addWidget(self.target_column)

        self.feature_selection_strategy = QComboBox()
        self.feature_selection_strategy.addItems(['SelectKBest (f_classif)', 'SelectKBest (mutual_info_classif)'])
        layout.addWidget(QLabel('Feature Selection Method:'))
        layout.addWidget(self.feature_selection_strategy)

        self.k_best_features = QSpinBox()
        self.k_best_features.setRange(1, 100)
        self.k_best_features.setValue(10)
        layout.addWidget(QLabel('Number of Best Features:'))
        layout.addWidget(self.k_best_features)

        tab.setLayout(layout)
        return tab

    def create_visualization_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.visualization_type = QComboBox()
        self.visualization_type.addItems(['Histogram', 'Scatter Plot'])
        layout.addWidget(QLabel('Visualization Type:'))
        layout.addWidget(self.visualization_type)

        self.x_column = QLineEdit()
        self.x_column.setPlaceholderText('X-axis Column (for Scatter Plot)')
        layout.addWidget(QLabel('X-axis Column:'))
        layout.addWidget(self.x_column)

        self.y_column = QLineEdit()
        self.y_column.setPlaceholderText('Y-axis Column (for Scatter Plot)')
        layout.addWidget(QLabel('Y-axis Column:'))
        layout.addWidget(self.y_column)

        self.plot_button = QPushButton('Plot')
        self.plot_button.clicked.connect(self.plot_data)
        layout.addWidget(self.plot_button)

        tab.setLayout(layout)
        return tab

    def create_outlier_detection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        self.outlier_strategy = QComboBox()
        self.outlier_strategy.addItems(['Z-Score', 'IQR'])
        layout.addWidget(QLabel('Outlier Detection Method:'))
        layout.addWidget(self.outlier_strategy)

        self.outlier_threshold = QLineEdit()
        self.outlier_threshold.setPlaceholderText('Threshold (e.g., 3 for Z-Score, 1.5 for IQR)')
        layout.addWidget(QLabel('Outlier Threshold:'))
        layout.addWidget(self.outlier_threshold)

        tab.setLayout(layout)
        return tab

    def load_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open CSV', '', 'CSV files (*.csv)')
        if file_name:
            self.df = pd.read_csv(file_name)
            self.update_column_list()
            self.update_preview()

    def update_column_list(self):
        self.column_list.clear()
        self.column_list.addItems(self.df.columns)

    def update_preview(self):
        if self.df is not None:
            self.preview.setText(self.df.head().to_string())

    def apply_changes(self):
        if self.df is None:
            QMessageBox.warning(self, 'No Data', 'Please load a CSV file first.')
            return

        selected_columns = [item.text() for item in self.column_list.selectedItems()]
        if not selected_columns:
            QMessageBox.warning(self, 'No Columns', 'Please select columns to process.')
            return

        current_tab = self.tabs.currentWidget()
        
        if current_tab == self.tabs.widget(0):  # Missing Values
            self.handle_missing_values(selected_columns)
        elif current_tab == self.tabs.widget(1):  # Encoding
            self.handle_encoding(selected_columns)
        elif current_tab == self.tabs.widget(2):  # Scaling
            self.handle_scaling(selected_columns)
        elif current_tab == self.tabs.widget(3):  # Binning
            self.handle_binning(selected_columns)
        elif current_tab == self.tabs.widget(4):  # Transformation
            self.handle_transformation(selected_columns)
        elif current_tab == self.tabs.widget(5):  # Feature Selection
            self.handle_feature_selection()
        elif current_tab == self.tabs.widget(6):  # Visualization
            self.plot_data()
        elif current_tab == self.tabs.widget(7):  # Outliers
            self.handle_outliers(selected_columns)

        self.update_preview()

    def handle_missing_values(self, selected_columns):
        strategy = self.missing_strategy.currentText()
        if strategy == 'Drop':
            self.df.dropna(subset=selected_columns, inplace=True)
        elif strategy in ['Fill with Mean', 'Fill with Median', 'Fill with Mode', 'Fill with Most Frequent']:
            fill_value = None
            if strategy == 'Fill with Mean':
                fill_value = self.df[selected_columns].mean()
            elif strategy == 'Fill with Median':
                fill_value = self.df[selected_columns].median()
            elif strategy == 'Fill with Mode':
                fill_value = self.df[selected_columns].mode().iloc[0]
            elif strategy == 'Fill with Most Frequent':
                fill_value = SimpleImputer(strategy='most_frequent')
                self.df[selected_columns] = fill_value.fit_transform(self.df[selected_columns])
            self.df[selected_columns] = self.df[selected_columns].fillna(fill_value)
        elif strategy == 'Fill with Value':
            fill_value = self.fill_value.text()
            self.df[selected_columns] = self.df[selected_columns].fillna(fill_value)

    def handle_encoding(self, selected_columns):
        strategy = self.encoding_strategy.currentText()
        if strategy == 'Label Encoding':
            encoder = LabelEncoder()
            for col in selected_columns:
                self.df[col] = encoder.fit_transform(self.df[col])
        elif strategy == 'One-Hot Encoding':
            self.df = pd.get_dummies(self.df, columns=selected_columns)

    def handle_scaling(self, selected_columns):
        strategy = self.scaling_strategy.currentText()
        if strategy == 'StandardScaler':
            scaler = StandardScaler()
        elif strategy == 'MinMaxScaler':
            scaler = MinMaxScaler()
        self.df[selected_columns] = scaler.fit_transform(self.df[selected_columns])

    def handle_binning(self, selected_columns):
        strategy = self.bin_strategy.currentText()
        num_bins = self.num_bins.value()
        for col in selected_columns:
            if strategy == 'Equal Width':
                self.df[col+'_binned'] = pd.cut(self.df[col], bins=num_bins)
            elif strategy == 'Equal Frequency':
                self.df[col+'_binned'] = pd.qcut(self.df[col], q=num_bins)

    def handle_transformation(self, selected_columns):
        strategy = self.transform_strategy.currentText()
        for col in selected_columns:
            if strategy == 'Log':
                self.df[col] = np.log1p(self.df[col])
            elif strategy == 'Square Root':
                self.df[col] = np.sqrt(self.df[col])
            elif strategy == 'Box-Cox':
                self.df[col], _ = stats.boxcox(self.df[col] + 1)

    def handle_feature_selection(self):
        target = self.target_column.text()
        if target not in self.df.columns:
            QMessageBox.warning(self, 'Invalid Target', 'The target column does not exist in the dataset.')
            return

        strategy = self.feature_selection_strategy.currentText()
        k = self.k_best_features.value()
        X = self.df.drop(columns=[target])
        y = self.df[target]
        
        if strategy == 'SelectKBest (f_classif)':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif strategy == 'SelectKBest (mutual_info_classif)':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        self.df = self.df[selected_features.tolist() + [target]]

    def handle_outliers(self, selected_columns):
        strategy = self.outlier_strategy.currentText()
        threshold = float(self.outlier_threshold.text())
        for col in selected_columns:
            if strategy == 'Z-Score':
                z_scores = np.abs(stats.zscore(self.df[col]))
                self.df = self.df[z_scores < threshold]
            elif strategy == 'IQR':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[~((self.df[col] < (Q1 - threshold * IQR)) | (self.df[col] > (Q3 + threshold * IQR)))]

    def plot_data(self):
        if self.df is None:
            QMessageBox.warning(self, 'No Data', 'Please load a CSV file first.')
            return

        plot_type = self.visualization_type.currentText()
        if plot_type == 'Histogram':
            self.df.hist(figsize=(10, 10))
        elif plot_type == 'Scatter Plot':
            x_col = self.x_column.text()
            y_col = self.y_column.text()
            if x_col not in self.df.columns or y_col not in self.df.columns:
                QMessageBox.warning(self, 'Invalid Columns', 'The specified columns do not exist in the dataset.')
                return
            self.df.plot(kind='scatter', x=x_col, y=y_col)

        plt.show()

    def save_csv(self):
        if self.df is None:
            QMessageBox.warning(self, 'No Data', 'Please load a CSV file first.')
            return

        file_name, _ = QFileDialog.getSaveFileName(self, 'Save CSV', '', 'CSV files (*.csv)')
        if file_name:
            self.df.to_csv(file_name, index=False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FeatureEngineeringApp()
    ex.show()
    sys.exit(app.exec_())
