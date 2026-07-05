create database data;
use data;
CREATE TABLE Customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(50),
    city VARCHAR(50),
    email VARCHAR(100)
);

INSERT INTO Customers VALUES
(1,'Amit Sharma','Delhi','amit@gmail.com'),
(2,'Priya Singh','Mumbai','priya@gmail.com'),
(3,'Rahul Verma','Pune','rahul@gmail.com'),
(4,'Sneha Gupta','Jaipur','sneha@gmail.com'),
(5,'Rohit Kumar','Lucknow','rohit@gmail.com'),
(6,'Anjali Mehta','Delhi','anjali@gmail.com'),
(7,'Vikas Jain','Indore','vikas@gmail.com'),
(8,'Neha Yadav','Kanpur','neha@gmail.com'),
(9,'Arjun Patel','Ahmedabad','arjun@gmail.com'),
(10,'Karan Malhotra','Chandigarh','karan@gmail.com');

CREATE TABLE Products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(50),
    category VARCHAR(30),
    price DECIMAL(10,2)
);

INSERT INTO Products VALUES
(101,'Laptop','Electronics',55000),
(102,'Mobile','Electronics',25000),
(103,'Headphones','Accessories',2000),
(104,'Keyboard','Accessories',1500),
(105,'Mouse','Accessories',800),
(106,'Tablet','Electronics',30000),
(107,'Monitor','Electronics',12000),
(108,'Printer','Office',7000),
(109,'Camera','Electronics',45000),
(110,'Speaker','Accessories',3500);

CREATE TABLE Salespersons (
    salesperson_id INT PRIMARY KEY,
    salesperson_name VARCHAR(50),
    region VARCHAR(30)
);

INSERT INTO Salespersons VALUES
(1,'Rajesh','North'),
(2,'Pooja','West'),
(3,'Manish','South'),
(4,'Deepak','East'),
(5,'Nisha','North'),
(6,'Rakesh','West'),
(7,'Simran','South'),
(8,'Ajay','East'),
(9,'Kavita','North'),
(10,'Mohit','West');

CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    salesperson_id INT,
    order_date DATE,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id),
    FOREIGN KEY (salesperson_id) REFERENCES Salespersons(salesperson_id)
);

INSERT INTO Orders VALUES
(1001,1,1,'2025-01-05'),
(1002,2,2,'2025-01-10'),
(1003,3,3,'2025-01-12'),
(1004,4,4,'2025-01-15'),
(1005,5,5,'2025-01-18'),
(1006,6,6,'2025-01-20'),
(1007,7,7,'2025-01-22'),
(1008,8,8,'2025-01-25'),
(1009,9,9,'2025-01-27'),
(1010,10,10,'2025-01-30');

CREATE TABLE Order_Details (
    order_detail_id INT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT,
    FOREIGN KEY (order_id) REFERENCES Orders(order_id),
    FOREIGN KEY (product_id) REFERENCES Products(product_id)
);

INSERT INTO Order_Details VALUES
(1,1001,101,1),
(2,1002,102,2),
(3,1003,103,3),
(4,1004,104,2),
(5,1005,105,5),
(6,1006,106,1),
(7,1007,107,2),
(8,1008,108,1),
(9,1009,109,1),
(10,1010,110,4);

select * from customers;
select * from products order by price desc limit 1 ;
select avg(price) as avg_price from products;
select count(*) as total_customers from customers;
select count(quantity) as qty_sold from order_details;
select min(price) lowest_price from products;
select max(price) highest_price from products;
select sum(price) as total_sales from products;
select c.customer_name , o.order_id, p.product_name from customers c join orders o on c.customer_id = o.customer_id join order_details od on o.order_id = 
od.order_id join products p on od.product_id = p.product_id;
select c.customer_name , o.order_id , c.city from orders o join customers c on o.customer_id = c.customer_id;
select city, count(*) as total_customer from customers group by city;
select p.category, count(od.quantity) as qty_sold from products p join order_details od  
on p.product_id = od.product_id group by p.category;
select s.salesperson_name , count(o.order_id) as total_orders 
from salespersons s join orders o on s.salesperson_id = o.salesperson_id group by salesperson_name;
select category, avg(price) as avg_price from products group by category;
