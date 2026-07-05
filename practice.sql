create database start;
use start;
CREATE TABLE Customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100),
    gender VARCHAR(10),
    city VARCHAR(50),
    state VARCHAR(50),
    signup_date DATE
);
INSERT INTO Customers VALUES
(1,'Amit Sharma','Male','Delhi','Delhi','2024-01-10'),
(2,'Priya Verma','Female','Mumbai','Maharashtra','2024-01-15'),
(3,'Rahul Singh','Male','Lucknow','UP','2024-02-01'),
(4,'Neha Gupta','Female','Delhi','Delhi','2024-02-10'),
(5,'Vikas Kumar','Male','Jaipur','Rajasthan','2024-02-20'),
(6,'Sneha Jain','Female','Pune','Maharashtra','2024-03-05'),
(7,'Rohit Mehta','Male','Chandigarh','Punjab','2024-03-12'),
(8,'Anjali Yadav','Female','Kanpur','UP','2024-03-18'),
(9,'Karan Patel','Male','Ahmedabad','Gujarat','2024-04-01'),
(10,'Pooja Sharma','Female','Bhopal','MP','2024-04-10');
CREATE TABLE Products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2)
);
INSERT INTO Products VALUES
(101,'Laptop','Electronics',55000),
(102,'Smartphone','Electronics',25000),
(103,'Headphones','Electronics',3000),
(104,'Office Chair','Furniture',7000),
(105,'Study Table','Furniture',12000),
(106,'Notebook','Stationery',100),
(107,'Pen Pack','Stationery',250),
(108,'Backpack','Accessories',1500),
(109,'Smart Watch','Electronics',5000),
(110,'Water Bottle','Accessories',500);
CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);
INSERT INTO Orders VALUES
(1001,1,'2024-04-01',58000),
(1002,2,'2024-04-02',25000),
(1003,3,'2024-04-03',7300),
(1004,4,'2024-04-05',12000),
(1005,5,'2024-04-08',3000),
(1006,6,'2024-04-10',5000),
(1007,7,'2024-04-12',1500),
(1008,8,'2024-04-15',600),
(1009,9,'2024-04-18',55000),
(1010,10,'2024-04-20',25250);
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
(2,1001,103,1),
(3,1002,102,1),
(4,1003,104,1),
(5,1003,107,1),
(6,1004,105,1),
(7,1005,103,1),
(8,1006,109,1),
(9,1007,108,1),
(10,1008,110,1),
(11,1009,101,1),
(12,1010,102,1),
(13,1010,107,1);

select customer_name from customers;
select product_name from products;
select * from products where price >5000;
select customer_name from customers where city = 'Delhi' ;
select count(*) as total_customer from customers;
select count(*) as total_products from products;
select * from products order by price desc limit 1; /* highest pice formula*/
select * from products order by price asc limit 1; /*lowest price fromula*/
select sum(total_amount) as total_sales from orders;
select avg(total_amount) as average_value from orders;
select sum(quantity) as total_sold from order_details;

SELECT p.category,
       SUM(p.price * od.quantity) AS revenue
FROM Products p
JOIN Order_Details od
ON p.product_id = od.product_id
GROUP BY p.category;

select c.city,
sum(o.total_amount) as revenue
from customers c
join orders o
on c.customer_id = o.customer_id
group by c.city;

select city ,
count(*) as total_customer from customers group by city;

select category ,
count(*) as total_category  from products group by category;

select customer_id ,
count(*) as total_orders from orders group by customer_id;

select product_id,
sum(quantity) as quantity_sold from order_details group by product_id;

select c.customer_name,
o.order_date 
from customers c
join orders o
on c.customer_id = o.customer_id;

select p.product_name,
sum(od.quantity) as quantity_sold
from products p
join order_details od
on p.product_id = od.product_id
group by p.product_name;

select c.customer_name,
p.product_name,
od.quantity
from customers c
join orders o
on c.customer_id = o.customer_id
join order_details od
on o.order_id = od.order_id
join products p
on od.product_id = p.product_id;

select c.customer_name,
sum(o.total_amount) as total_spending
from customers c
join orders o 
on c.customer_id = o.customer_id
group by c.customer_name;

select c.customer_name,
sum(o.total_amount) as total_spending
from customers c
join orders o 
on c.customer_id = o.customer_id
group by c.customer_name
order by total_spending desc limit 3;

select p.product_name,
sum(od.quantity) as total_sold
from products p
join order_details od
on p.product_id  = od.product_id
group by p.product_name
order by total_sold desc limit 1;

select p.product_name
from products p 
left join order_details od
on p.product_id = od.product_id
where od.product_id is null;

select c.customer_name
from customers c
left join orders o
on c.customer_id = o.customer_id
where o.customer_id is null;

select c.customer_name,
sum(o.total_amount) as total_spending,
rank() over(order by sum(o.total_amount) desc ) as ranking 
from customers c
join orders o
on c.customer_id = o.customer_id
group by c.customer_name;

select customer_name,
total_spending
from
 (select c.customer_name,
 sum(o.total_amount) as total_spending,
 dense_rank() over( 
 order by sum(o.total_amount) desc
 ) as rnk
 from customers c
 join orders o
 on c.customer_id = o.customer_id
 group by c.customer_name
 )t
 where rnk = 2;
 
 
 
 
 
 
 
 

