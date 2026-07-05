create database company;
use company;
CREATE TABLE Departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50),
    location VARCHAR(50)
);

INSERT INTO Departments VALUES
(1,'HR','Delhi'),
(2,'IT','Mumbai'),
(3,'Finance','Bangalore'),
(4,'Sales','Pune'),
(5,'Marketing','Chennai'),
(6,'Operations','Hyderabad'),
(7,'Support','Kolkata'),
(8,'Research','Noida'),
(9,'Admin','Jaipur'),
(10,'Legal','Ahmedabad');

CREATE TABLE Employees (
    emp_id INT PRIMARY KEY,
    emp_name VARCHAR(50),
    gender VARCHAR(10),
    salary DECIMAL(10,2),
    hire_date DATE,
    dept_id INT,
    FOREIGN KEY (dept_id) REFERENCES Departments(dept_id)
);

INSERT INTO Employees VALUES
(101,'Amit','Male',50000,'2022-01-15',2),
(102,'Priya','Female',60000,'2021-03-10',1),
(103,'Rahul','Male',75000,'2020-06-20',3),
(104,'Sneha','Female',45000,'2023-02-11',4),
(105,'Vikas','Male',55000,'2021-08-18',5),
(106,'Neha','Female',70000,'2019-05-25',2),
(107,'Arjun','Male',65000,'2022-09-01',6),
(108,'Pooja','Female',48000,'2023-04-10',7),
(109,'Karan','Male',80000,'2018-11-15',8),
(110,'Anjali','Female',62000,'2020-12-05',9),
(111,'Rohit','Male',58000,'2022-07-20',4),
(112,'Meera','Female',72000,'2021-01-15',3);

CREATE TABLE Projects (
    project_id INT PRIMARY KEY,
    project_name VARCHAR(100),
    budget DECIMAL(12,2),
    start_date DATE
);

INSERT INTO Projects VALUES
(201,'ERP System',500000,'2023-01-01'),
(202,'Website Redesign',200000,'2023-02-15'),
(203,'Mobile App',350000,'2023-03-01'),
(204,'AI Chatbot',400000,'2023-04-10'),
(205,'CRM Upgrade',250000,'2023-05-20'),
(206,'Cloud Migration',600000,'2023-06-01'),
(207,'Data Warehouse',700000,'2023-07-01'),
(208,'Cyber Security',450000,'2023-08-15'),
(209,'E-Commerce Platform',550000,'2023-09-01'),
(210,'HR Automation',150000,'2023-10-01');

CREATE TABLE Employee_Projects (
    emp_id INT,
    project_id INT,
    hours_worked INT,
    PRIMARY KEY(emp_id, project_id),
    FOREIGN KEY(emp_id) REFERENCES Employees(emp_id),
    FOREIGN KEY(project_id) REFERENCES Projects(project_id)
);

INSERT INTO Employee_Projects VALUES
(101,201,120),
(101,202,80),
(102,210,60),
(103,207,150),
(104,205,90),
(105,209,110),
(106,204,140),
(107,206,130),
(108,208,70),
(109,207,180),
(110,210,85),
(111,205,95),
(112,203,125),
(103,203,75),
(106,206,100);

/* Questions*/
select * from employees;
select emp_name,salary from employees;
select * from employees where salary > 60000;
select * from employees where gender = 'female';
select * from employees where hire_date >'2021-01-01';
select dept_name from departments;
select * from projects where budget > 400000;
select * from employees order by salary desc;
select * from employees order by salary desc limit 5;
select count(*) as total_employees from employees;
select avg(salary) as avg_salary from employees;
select min(salary) as lowest_salary,max(salary) as highest_salary from employees;
select dept_id , count(*) as emp_count from employees group by dept_id;
select dept_id , count(*) as empl_count from employees group by dept_id having count(*)>1;
select sum(budget) as total_budget from projects;
select * from employees where salary > (select avg(salary) from employees);
select * from employees where emp_name like 'A%';
select * from employees where emp_name like '%a%';

select e.emp_name,
d.dept_name 
from employees e 
join departments d 
on e.dept_id = d.dept_id;

select emp_id,
sum(hours_worked) as total_hours from employee_projects group by emp_id;

select e.emp_name,
d.dept_name
from employees e inner join departments d 
on e.dept_id = d.dept_id;

select e.emp_name, 
p.project_name
from employees e join employee_projects ep
on e.emp_id = ep.emp_id
join projects p 
on ep.project_id = p.project_id;

select e. * 
from employees e 
left join employee_projects ep 
on e.emp_id = ep.emp_id 
where ep.emp_id is null;

select p. * 
from projects p 
left join employee_projects ep
on p.project_id = ep.project_id
where ep.project_id is null;

select d.dept_name,
avg(e.salary) as avg_salary
from employees e 
join departments d
on e.dept_id = d.dept_id
group by d.dept_name;

select d.dept_name,
avg(e.salary) as avg_salary 
from employees e
join departments d 
on e.dept_id = d.dept_id
group by d.dept_id
order by avg_salary desc limit 1;

select emp_name, salary ,
rank() over(order by salary desc) as salary_rank from employees;

select max(salary) from employees where salary < (select max(salary) from employees);

SELECT e.emp_name,
       SUM(ep.hours_worked) AS total_hours
FROM Employees e
JOIN Employee_Projects ep
ON e.emp_id = ep.emp_id
GROUP BY e.emp_name
ORDER BY total_hours DESC
LIMIT 1;

select * from projects order by budget desc limit 3;

select emp_id ,
count(project_id) as project_count
from employee_projects
group by emp_id
having count(project_id) >1;

select emp_name , salary,
sum(salary) over(order by salary) as commulative_salary from employees;

select * from (select e. * , rank() over(partition by dept_id order by salary desc) rnk from employees e) x where rnk=1;

select e.*
from employees e where salary > (select avg(salary) from employees  where dept_id = e.dept_id);

select d.dept_name,
avg(e.salary) as avg_salary
from employees e 
join departments d
on e.dept_id = d.dept_id
group by dept_name
having avg(e.salary) > 60000;

