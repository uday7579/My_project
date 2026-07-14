create database swiggy;
use swiggy;
CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY,
    CustomerName VARCHAR(50),
    City VARCHAR(50),
    SignupDate DATE
);

INSERT INTO Customers VALUES
(1,'Rahul Sharma','Delhi','2024-01-10'),
(2,'Priya Singh','Mumbai','2024-02-15'),
(3,'Aman Verma','Bangalore','2024-03-01'),
(4,'Neha Gupta','Delhi','2024-01-20'),
(5,'Rohit Kumar','Pune','2024-04-05'),
(6,'Sneha Jain','Hyderabad','2024-03-18'),
(7,'Karan Patel','Ahmedabad','2024-02-10'),
(8,'Anjali Mehta','Chennai','2024-05-01'),
(9,'Vikas Yadav','Lucknow','2024-04-12'),
(10,'Pooja Mishra','Jaipur','2024-06-01');

CREATE TABLE Restaurants (
    RestaurantID INT PRIMARY KEY,
    RestaurantName VARCHAR(100),
    City VARCHAR(50),
    Cuisine VARCHAR(50),
    Rating DECIMAL(2,1)
);

INSERT INTO Restaurants VALUES
(101,'Burger Hub','Delhi','Fast Food',4.3),
(102,'Pizza Point','Mumbai','Italian',4.5),
(103,'Biryani House','Bangalore','Indian',4.7),
(104,'South Spice','Chennai','South Indian',4.4),
(105,'Punjabi Tadka','Delhi','North Indian',4.6),
(106,'Roll Express','Pune','Fast Food',4.2),
(107,'Chinese Wok','Hyderabad','Chinese',4.1),
(108,'Cafe Coffee Day','Bangalore','Cafe',4.0),
(109,'Dominos','Jaipur','Pizza',4.3),
(110,'KFC','Lucknow','Fast Food',4.4);

CREATE TABLE Orders (
    OrderID INT PRIMARY KEY,
    CustomerID INT,
    RestaurantID INT,
    OrderDate DATE,
    OrderAmount DECIMAL(10,2),
    DeliveryFee DECIMAL(10,2),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
    FOREIGN KEY (RestaurantID) REFERENCES Restaurants(RestaurantID)
);

INSERT INTO Orders VALUES
(1001,1,101,'2025-06-01',350,40),
(1002,2,102,'2025-06-02',550,50),
(1003,3,103,'2025-06-02',420,35),
(1004,4,105,'2025-06-03',600,45),
(1005,5,106,'2025-06-04',280,30),
(1006,6,107,'2025-06-05',500,40),
(1007,7,108,'2025-06-06',220,25),
(1008,8,104,'2025-06-07',650,50),
(1009,9,110,'2025-06-08',400,35),
(1010,10,109,'2025-06-09',700,55),
(1011,1,103,'2025-06-10',480,40),
(1012,2,105,'2025-06-11',750,50);

CREATE TABLE DeliveryPartners (
    DeliveryID INT PRIMARY KEY,
    OrderID INT,
    PartnerName VARCHAR(50),
    DeliveryTime INT,
    DeliveryStatus VARCHAR(20),
    FOREIGN KEY (OrderID) REFERENCES Orders(OrderID)
);

INSERT INTO DeliveryPartners VALUES
(1,1001,'Rakesh',28,'Delivered'),
(2,1002,'Suresh',32,'Delivered'),
(3,1003,'Mohit',25,'Delivered'),
(4,1004,'Ankit',35,'Delivered'),
(5,1005,'Deepak',22,'Delivered'),
(6,1006,'Vivek',30,'Delivered'),
(7,1007,'Ajay',18,'Delivered'),
(8,1008,'Manish',40,'Delayed'),
(9,1009,'Rohit',26,'Delivered'),
(10,1010,'Kunal',38,'Delayed'),
(11,1011,'Rakesh',29,'Delivered'),
(12,1012,'Suresh',34,'Delivered');

select count(orderid) as total_orders from orders;
select avg(orderamount) as avg_order from orders;
select* from orders order by orderamount desc limit 5;
select count(*) as total_customers from customers group by city order by total_customers desc;
select r.restaurantname , sum(o.orderamount) as revenue from orders o join restaurants r on o.restaurantid = r.restaurantid 
group by restaurantname order by revenue desc;
select r.restaurantname , count(o.orderid) as total_orders from orders o join restaurants r on r.restaurantid = o.restaurantid 
group by r.restaurantname order by total_orders desc;
select partnername, round(avg(deliverytime),2) as avg_deivery_time from deliverypartners group by partnername order by avg_deivery_time desc;
select c.customername , count(o.orderid) as order_count from customers c join orders o on c.customerid = o.customerid group by customername order by order_count;
select c.city, sum(o.orderamount) as total_orders from orders o join customers c on c.customerid = o.customerid group by city order by total_orders ;
select c.customername , sum(o.orderamount) as total_orders from orders o join customers c on c.customerid = o.customerid group by customername order by total_orders ;
select r.restaurantname , r.rating , sum(o.orderamount) as total_revenue from restaurants r join orders o on r.restaurantid = o.restaurantid 
group by restaurantname, r.rating order by total_revenue desc;
select * from deliverypartners where deliverystatus = 'delayed';
select r.restaurantname , sum(o.orderamount) as revenue, rank() over(order by sum(o.orderamount) desc ) as rnk from orders o join restaurants r 
group by restaurantname;
select r.restaurantname , sum(o.orderamount) as revenue , dense_rank() over(order by sum(o.orderamount) desc ) as rnk from orders o join 
restaurants r on o.restaurantid = r.restaurantid group by r.restaurantname;
select orderdate, sum(orderamount) as daily_sales , sum(sum(orderamount)) over(order by orderdate ) as running_total from orders o group by orderdate;
select customername,totalspent from ( select c.customername , sum(o.orderamount) as totalspent, dense_rank() over( order by sum(o.orderamount) desc ) as rnk 
from customers c join orders o 