# Customer segmmentation 
# Get delivery customers 
df_delicust = df[df.IsDelivery == 1]

# Change dtype to datetime
df_delicust.InvoiceDateHour = pd.to_datetime(df_delicust.InvoiceDateHour, format='%Y-%m-%d %H:%M:%S.%f')

# Mark order as dinner meal if invoice date after 17
df_delicust["Dinner"] = [0 if x.hour < 17 else 1 for x in df_delicust.InvoiceDateHour]
df_delicust.Dinner.value_counts()

# Create df with orders
agg_func = {"DocNumber": "first", "ProductDesignation": lambda x: ",".join(x) , "ProductFamily": lambda x: ",".join(x), 
                          "Qty":"sum", "TotalAmount":"sum", "InvoiceDateHour": "first", "CustomerID": "first", "Dinner": "first"}

deliorders = df_delicust[["DocNumber", "ProductDesignation", "ProductFamily", 
                          "Qty", "TotalAmount", "InvoiceDateHour", "CustomerID", "Dinner"]].groupby("DocNumber").aggregate(agg_func)

deliorders.Dinner.value_counts()


# Create df with customers
agg_func = {"DocNumber": "count", "ProductDesignation": lambda x: ",".join(x) , "ProductFamily": lambda x: ",".join(x), 
                          "Qty":"sum", "TotalAmount":"sum", "InvoiceDateHour": "first", "Dinner": "count"}

delicust = deliorders[["DocNumber", "ProductDesignation", "ProductFamily", 
                          "Qty", "TotalAmount", "InvoiceDateHour", "CustomerID", "Dinner"]].groupby("CustomerID").aggregate(agg_func)
delicust["Lunch"] = delicust["DocNumber"] - delicust["Dinner"]


# Kmeans based on number of vists during dinner and lunch, total spendings and quantity
X_model = delicust[[ "Qty", "TotalAmount", "Dinner", "Lunch"]].copy(deep=True)
# normalize variables 
min_max_scaler = MinMaxScaler() 
X_model_scaled = min_max_scaler.fit_transform(X_model) 
X_model_norm = pd.DataFrame(X_model_scaled, columns=X_model.columns) 

# Select K based on the sum of squared distances 
ssd = [] 

K = range(1,12) 

for k in K: 
    km = KMeans(n_clusters=k, random_state=145) 
    km = km.fit(X_model_norm) 
    ssd.append(km.inertia_) 

# Plot results in an elbow graph 
plt.plot(K, ssd, 'bx-') 
plt.xlabel('number of K') 
plt.ylabel('Sum of squared distances') 
plt.title('Elbow method - Reduced dimensionality') 
plt.show() 


# Apply the K-Means for K= 3
K= 3

kmeans = KMeans(n_clusters=K, random_state=145) 

kmeans.fit(X_model_norm) 

y_kmeans = kmeans.predict(X_model_norm) 

X_model["Cluster"]=y_kmeans


X_model.Cluster.value_counts()


