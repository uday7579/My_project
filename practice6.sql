create database zomato;
use zomato;
CREATE TABLE Customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(50),
    city VARCHAR(30),
    gender VARCHAR(10),
    age INT
);

INSERT INTO Customers VALUES
(1,'Rahul','Delhi','Male',24),
(2,'Priya','Mumbai','Female',27),
(3,'Aman','Pune','Male',30),
(4,'Neha','Jaipur','Female',22),
(5,'Rohit','Lucknow','Male',29),
(6,'Anjali','Delhi','Female',26),
(7,'Karan','Noida','Male',31),
(8,'Sneha','Hyderabad','Female',28),
(9,'Vikas','Chandigarh','Male',35),
(10,'Pooja','Indore','Female',25);

CREATE TABLE Restaurants (
    restaurant_id INT PRIMARY KEY,
    restaurant_name VARCHAR(50),
    city VARCHAR(30),
    cuisine VARCHAR(30),
    rating DECIMAL(2,1)
);

INSERT INTO Restaurants VALUES
(101,'Burger Point','Delhi','Fast Food',4.2),
(102,'Pizza Hub','Mumbai','Italian',4.5),
(103,'Spice Villa','Pune','North Indian',4.0),
(104,'Chinese Wok','Jaipur','Chinese',3.8),
(105,'Biryani House','Lucknow','Biryani',4.7),
(106,'Tandoori Treat','Delhi','North Indian',4.4),
(107,'Food Corner','Noida','Fast Food',3.9),
(108,'Dosa Plaza','Hyderabad','South Indian',4.3),
(109,'BBQ Nation','Chandigarh','BBQ',4.6),
(110,'Cafe Coffee','Indore','Cafe',4.1);

CREATE TABLE Orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    restaurant_id INT,
    order_date DATE,
    amount DECIMAL(8,2),
    payment_method VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id),
    FOREIGN KEY (restaurant_id) REFERENCES Restaurants(restaurant_id)
);

INSERT INTO Orders VALUES
(1001,1,101,'2025-01-05',450,'UPI'),
(1002,2,102,'2025-01-08',780,'Card'),
(1003,3,103,'2025-01-09',650,'Cash'),
(1004,4,104,'2025-01-12',320,'UPI'),
(1005,5,105,'2025-01-15',900,'Card'),
(1006,6,106,'2025-01-18',540,'UPI'),
(1007,7,107,'2025-01-19',410,'Cash'),
(1008,8,108,'2025-01-20',350,'UPI'),
(1009,9,109,'2025-01-22',1200,'Card'),
(1010,10,110,'2025-01-25',280,'Cash');

CREATE TABLE Delivery (
    delivery_id INT PRIMARY KEY,
    order_id INT,
    delivery_partner VARCHAR(50),
    delivery_time INT,
    delivery_status VARCHAR(20),
    FOREIGN KEY (order_id) REFERENCES Orders(order_id)
);

INSERT INTO Delivery VALUES
(201,1001,'Amit',25,'Delivered'),
(202,1002,'Rakesh',35,'Delivered'),
(203,1003,'Sanjay',42,'Delivered'),
(204,1004,'Vijay',30,'Cancelled'),
(205,1005,'Ajay',28,'Delivered'),
(206,1006,'Deepak',20,'Delivered'),
(207,1007,'Mohit',33,'Delivered'),
(208,1008,'Ravi',22,'Delivered'),
(209,1009,'Ankit',40,'Delivered'),
(210,1010,'Sumit',18,'Delivered');

--- questions----
select * from restaurants;
select * from customers where city = 'Delhi';
select max(amount) as highest_order from orders;
select min(amount) as highest_order from orders;
select (payment_method) from orders;
select c.customer_name , o.amount as order_amount from customers c join orders o on c.customer_id = o.customer_id;
select r.restaurant_name , o.amount as order_amount from restaurants r join orders o on r.restaurant_id = o.restaurant_id;
select sum(amount) as revenue from orders;
select avg(amount) as order_value from orders;
select count(payment_method) as total_orders from orders;
select r.restaurant_name , sum(o.amount) as total_revenue from restaurants r join orders o on r.restaurant_id = o.restaurant_id group by r.restaurant_name;
select cuisine , avg(rating) as avg_rating from restaurants group by cuisine;
select c.customer_name , o.payment_method from customers c join orders o on c.customer_id = o.customer_id where payment_method = 'UPI';
select restaurant_name , avg(rating) as avg_rating from restaurants group by restaurant_name;
select restaurant_id , amount from orders order by amount desc limit 3;
select * from delivery where delivery_status = 'delivered';
select * from delivery where delivery_status = 'cancelled';
select city, count(*) as total_customers from customers group by city;
select c.customer_name , o.amount from customers c inner join orders o on c.customer_id = o.customer_id;
select o.order_id , r.restaurant_name from orders o inner join restaurants r on r.restaurant_id = o.restaurant_id;
select c.customer_name , o.amount , r.restaurant_name , d.delivery_status from customers c inner join orders o on
c.customer_id = o.customer_id join restaurants r on o.restaurant_id = r.restaurant_id join delivery d on 
o.order_id = d.order_id;
select c.customer_name , d.delivery_partner ,o.amount from customers c inner join orders o on
c.customer_id = o.customer_id join delivery d on o.order_id = d.order_id;
select c.city as customer_city , r.city as restaurant_city from customers c inner join orders o on
c.customer_id = o.customer_id join restaurants r on o.restaurant_id = r.restaurant_id;
select r.rating , o.order_id from restaurants r join orders o on o.restaurant_id = r.restaurant_id;
select c.customer_name , d.delivery_status from customers c inner join orders o on c.customer_id = o.customer_id
join delivery d on o.order_id = d.order_id;
select o.amount as order_amount , d.delivery_time from orders o
join delivery d on o.order_id = d.order_id;
select avg(d.delivery_time) as avg_delivery_time , r.restaurant_name from restaurants r inner join 
orders o on o.restaurant_id = r.restaurant_id join delivery d on  o.order_id = d.order_id group by restaurant_name;
select count(o.order_id) as total_orders , r.restaurant_name from orders o join restaurants r on
 o.restaurant_id = r.restaurant_id group by restaurant_name;
select sum(o.amount) as total_revenue , c.city from customers c join orders o on c.customer_id = o.customer_id group by city;
select sum(o.amount) as total_revenue ,r.restaurant_name, r.rating from restaurants r join orders o 
on r.restaurant_id = o.restaurant_id group by rating,restaurant_name;
select sum(amount) as total_revenue ,payment_method from orders group by payment_method;
select count(o.order_id) as total_orders ,r.city from orders o join restaurants r 
on o.restaurant_id = r.restaurant_id  group by r.city;
select avg(o.amount) as avg_amount , r.city from orders o join restaurants r 
on o.restaurant_id = r.restaurant_id  group by r.city;
select count(restaurant_id) total_restaurants , cuisine from restaurants  group by cuisine;
select sum(o.amount) revenue , r.cuisine from restaurants r join orders o on 
 o.restaurant_id = r.restaurant_id group by cuisine;